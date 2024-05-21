from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from .monitor import AverageMeter
from tqdm import tqdm
from ..data.utils import lengths_to_mask
from ..model.operator.clip_module import CLIPTextEncoder
from ..model.diffusion import Diffusion
from ..model.network import t2m_motionenc, t2m_textenc
from ..visual.plot_script import plot_3d_motion
from ..visual import paramUtil 
import numpy as np
from ..metrics.tm2t import TM2TMetrics
from ..metrics.gru import HUMANACTMetrics
from ..transforms.rotation2xyz import Rotation2xyz

condition_error = "condition must in ['text', 'action']"

@torch.no_grad()
def update_ema(target_params, source_params, rate=0.99):
    """
    Update target parameters to be closer to those of source parameters using
    an exponential moving average.

    :param target_params: the target parameter sequence.
    :param source_params: the source parameter sequence.
    :param rate: the EMA rate (closer to 1 means slower).
    """
    for targ, src in zip(target_params, source_params):
        targ.detach().mul_(rate).add_(src, alpha=1 - rate)

class MLCTTrainer(object):
    def __init__(self, cfg, logger) -> None:
        self.cfg = cfg
        self.logger = logger
        self.feats2joints = None
        self.device = cfg.device + str(cfg.gpu)
        self.condition = cfg.condition

        if self.condition == 'text':
            self.TM2TMetrics = TM2TMetrics(
                diversity_times=cfg.test.diversity_times
            )
        elif self.condition == 'action':
            self.HUMANACTMetrics = HUMANACTMetrics(
                        datapath=os.path.join('deps', "humanact12_gru.tar"),
                        diversity_times=self.cfg.test.diversity_times,
                        multimodality_times=self.cfg.test.mm_num_times,
                        dist_sync_on_step=True,
                    )
        else:
            assert 0, condition_error

        if cfg.condition == 'text':
            self._get_t2m_evaluator(cfg)
        elif cfg.condition == 'action':
            self._get_a2m_evaluator(cfg)
        else:
            assert 0

    def train(self, model, datamodule):
        cfg = self.cfg
        self.skeleton = paramUtil.kit_kinematic_chain if cfg.data.name == 'kit' else paramUtil.t2m_kinematic_chain

        # prepare model
        mae = model['mae']
        denoiser = model['denoiser']
        denoiser_ema = model['denoiser_ema']
        if self.condition == 'text':
            model['condition'] = CLIPTextEncoder(cfg.diffusion.clip_path, last_hidden_state=False).to(self.device)
        else:
            model['condition'] = nn.Embedding(num_embeddings=12, embedding_dim=768)
        mae = mae.to(self.device)
        denoiser = denoiser.to(self.device)
        denoiser_ema = denoiser_ema.to(self.device)
        
        # prepare
        diffusion = Diffusion(cfg)

        # optimizer
        if self.condition == 'text':
            params = [{'params': denoiser.parameters()}]
        elif self.condition == 'action':
            params = [{'params': denoiser.parameters()}, {'params': model['condition'].parameters()}]
        else:
            assert 0, "condition must in ['text', 'action']"
        optimizer = AdamW(lr=cfg.train.lr, params=params)
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, eta_min=cfg.train.lr_min, last_epoch=-1)

        # recover joints function
        if cfg.condition == 'text':
            self.feats2joints = datamodule.feats2joints
            self.renorm4t2m = datamodule.renorm4t2m
        elif cfg.condition == 'action':
            self.rot2xyz = Rotation2xyz(smpl_path=cfg.data.smpl_path)
            self.feats2joints_eval = lambda sample, mask: self.rot2xyz(
                sample.view(*sample.shape[:-1], 6, 25).permute(0, 3, 2, 1),
                mask=mask,
                pose_rep='rot6d',
                glob=True,
                translation=True,
                jointstype='smpl',
                vertstrans=True,
                betas=None,
                beta=0,
                glob_rot=None,
                get_rotations_back=False)
            self.feats2joints = lambda sample, mask: self.rot2xyz(
                sample.view(*sample.shape[:-1], 6, 25).permute(0, 3, 2, 1),
                mask=mask,
                pose_rep='rot6d',
                glob=True,
                translation=True,
                jointstype='vertices',
                vertstrans=False,
                betas=None,
                beta=0,
                glob_rot=None,
                get_rotations_back=False)
        else:
            assert 0, condition_error
            

        best_FID = 1000

        # dataloader
        train_dataloader = datamodule.train_dataloader()
        test_dataloader = datamodule.test_dataloader()

        for epoch in range(cfg.train.epochs):
            ema_rate = self.cfg.train.ema_rate

            # train step
            train_loss = self.train_step(model, optimizer, train_dataloader, diffusion, ema_rate, epoch=epoch)
            self.logger.info(f'Epoch {epoch} Train Loss {train_loss:.6f} EMA Rate {ema_rate}.')

            if (epoch + 1) % cfg.train.test_epochs == 0:
                FID_list = self.test_step(model, test_dataloader, diffusion, epoch)
                self.logger.info(f'Epoch {epoch + 1} FID List {FID_list} FID {np.mean(FID_list):.6f}.')

                with open(os.path.join(cfg.output_dir, 'FID.txt'), 'a') as f:
                    f.write(f'{epoch + 1},{np.mean(FID_list):.6f};\n')

                if np.mean(FID_list) < best_FID:
                    best_FID = np.mean(FID_list)
                    torch.save(model['denoiser_ema'].state_dict(), os.path.join(cfg.save_dir, f'Epoch_Best_FID.pth'))
                    self.logger.info(f'Save Best {np.mean(FID_list):.6f}.')

            # if (epoch + 1) % cfg.train.visual_epochs == 0:
            #     self.visual_step(model, test_dataloader, diffusion, epoch)

            if (epoch + 1) % cfg.train.save_epochs == 0:
                torch.save(model['denoiser_ema'].state_dict(), os.path.join(cfg.save_dir, f'Epoch_{epoch+1:04d}_loss_{train_loss:.4f}.pth'))
                if self.condition == 'action':
                    torch.save(model['condition'].state_dict(), os.path.join(cfg.save_dir, f'Epoch_{epoch+1:04d}_condition.pth'))

            lr_scheduler.step()

    def _get_t2m_evaluator(self, cfg):
        """
        load T2M text encoder and motion encoder for evaluating
        """
        # init module
        self.t2m_textencoder = t2m_textenc.TextEncoderBiGRUCo(
            word_size=cfg.eval.t2m_textencoder.dim_word,
            pos_size=cfg.eval.t2m_textencoder.dim_pos_ohot,
            hidden_size=cfg.eval.t2m_textencoder.dim_text_hidden,
            output_size=cfg.eval.t2m_textencoder.dim_coemb_hidden,
        ).to(self.device)

        self.t2m_moveencoder = t2m_motionenc.MovementConvEncoder(
            input_size=cfg.data.dim_pose - 4,
            hidden_size=cfg.eval.t2m_motionencoder.dim_move_hidden,
            output_size=cfg.eval.t2m_motionencoder.dim_move_latent,
        ).to(self.device)

        self.t2m_motionencoder = t2m_motionenc.MotionEncoderBiGRUCo(
            input_size=cfg.eval.t2m_motionencoder.dim_move_latent,
            hidden_size=cfg.eval.t2m_motionencoder.dim_motion_hidden,
            output_size=cfg.eval.t2m_motionencoder.dim_motion_latent,
        ).to(self.device)

        # load pretrianed
        dataname = cfg.data.name
        dataname = "t2m" if dataname == "humanml3d" else dataname
        t2m_checkpoint = torch.load(os.path.join('deps', dataname, "text_mot_match/model/finest.tar"), map_location=self.device)
        self.t2m_textencoder.load_state_dict(t2m_checkpoint["text_encoder"])
        self.t2m_moveencoder.load_state_dict(t2m_checkpoint["movement_encoder"])
        self.t2m_motionencoder.load_state_dict(t2m_checkpoint["motion_encoder"])

        # freeze params
        self.t2m_textencoder.eval()
        self.t2m_moveencoder.eval()
        self.t2m_motionencoder.eval()
        for p in self.t2m_textencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_moveencoder.parameters():
            p.requires_grad = False
        for p in self.t2m_motionencoder.parameters():
            p.requires_grad = False

    def _get_a2m_evaluator(self, cfg):
        pass

    def train_step(self, model, optimizer, train_dataloader, diffusion, ema_rate, epoch):
        mae = model['mae']
        denoiser = model['denoiser']
        condition_encoder = model['condition']
        denoiser_ema = model['denoiser_ema']
        loss_monitor = AverageMeter('loss')

        mae.eval()
        condition_encoder.eval()
        denoiser.train()
        denoiser_ema.eval()

        loop = tqdm((train_dataloader), total = len(train_dataloader))
        for batch in loop:
            optimizer.zero_grad()
            if self.condition == 'text':
                motion, cond, length = batch['motion'], batch['text'], batch['length']
            elif self.condition == 'action':
                motion, cond, length = batch['motion'], batch['action'], batch['length']
            else:
                assert 0, "condition must in ['text', 'action']"

            motion = motion.to(self.device)
            mask = lengths_to_mask(length, max_len=self.cfg.data.max_motion_length, device=self.device)

            with torch.no_grad():
                latent, _ = mae.motion_encode(motion, ~mask)
                latent_cond = condition_encoder(cond).to(self.device)

                if self.cfg.diffusion.uncod_type == 'trainable':
                    if self.condition == 'text':
                        latent_cond_none = condition_encoder(['']).repeat(motion.shape[0], 1, 1)
                    elif self.condition == 'action':
                        latent_cond_none = torch.zeros((motion.shape[0], 1, 768), device=self.device)
                    else:
                        assert 0, condition_error
                    
                else:
                    latent_cond_none = None

            loss = diffusion.cal_loss(latent, latent_cond, latent_cond_none, denoiser, denoiser_ema)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(parameters=denoiser.parameters(), max_norm=10, norm_type=2)
            optimizer.step()
            loss_monitor.update(loss.item(), n=motion.shape[0])

            loop.set_description(f'Train Epoch [{epoch+1}/{self.cfg.train.epochs}]')
            loop.set_postfix(loss = loss_monitor.avg)

            with torch.no_grad():
                update_ema(denoiser_ema.parameters(), denoiser.parameters(), rate=ema_rate)
            
        return loss_monitor.avg
    
    def visual_step(self, model, test_dataloader, diffusion, epoch):
        mae = model['mae']
        denoiser_ema = model['denoiser_ema']
        condition_encoder = model['condition']

        mae.eval()
        condition_encoder.eval()
        denoiser_ema.eval()

        for batch in test_dataloader:
            motion, text, length = batch['motion'], batch['text'], batch['length']
            motion = motion[:1]
            condition = text[:1]
            length = length[:1]
            break

        motion = motion.to(self.device)

        feats_rst = diffusion.sample(bs=1, lengths=length, condition=condition, steps=self.cfg.train.sample_steps, denoiser=denoiser_ema, motion_encoder=mae,
                                     condition_encoder=condition_encoder, device=self.device)
     
        fps = 12.5 if self.cfg.data.name == 'kit' else 20
        joints_pred = self.feats2joints(feats_rst)[0, :length[0]].detach().cpu().numpy()
        plot_3d_motion(os.path.join(self.cfg.sample_dir, f"sample_pred_motion_{epoch+1:04d}.gif"), self.skeleton, joints_pred, dataset=self.cfg.data.name, title=f'{condition[0]}', fps=fps)
            
        joints_gt = self.feats2joints(motion)[0, :length[0]].detach().cpu().numpy()
        plot_3d_motion(os.path.join(self.cfg.sample_dir, f"sample_gt_motion_{epoch+1:04d}.gif"), self.skeleton, joints_gt, dataset=self.cfg.data.name, title=f'{condition[0]}', fps=fps)
        
    @torch.no_grad()
    def test_step(self, model, test_dataloader, diffusion, epoch):
        if self.condition == 'text':
            mae = model['mae']
            denoiser_ema = model['denoiser_ema']
            condition_encoder = model['condition']
            mae.eval()
            denoiser_ema.eval()
            condition_encoder.eval()
            FID_list = []
            for repeat_idx in range(3):
                self.TM2TMetrics.init()

                loop = tqdm(enumerate(test_dataloader), total = len(test_dataloader))
                for idx, batch in loop:
                    texts = batch["text"]
                    motions = batch["motion"].detach().clone()
                    lengths = batch["length"]
                    word_embs = batch["word_embs"].detach().clone()
                    pos_ohot = batch["pos_ohot"].detach().clone()
                    text_lengths = batch["text_len"].detach().clone()

                    motions = motions.to(self.device)
                    word_embs = word_embs.to(self.device)
                    pos_ohot = pos_ohot.to(self.device)
                    text_lengths = text_lengths.to(self.device)

                    feats_rst = diffusion.sample(bs=len(lengths), lengths=lengths, condition=texts, steps=self.cfg.train.sample_steps, denoiser=denoiser_ema, motion_encoder=mae,
                                                condition_encoder=condition_encoder, device=self.device)
                    
                    # joints recover
                    joints_rst = self.feats2joints(feats_rst)
                    joints_ref = self.feats2joints(motions)

                    # renorm for t2m evaluators
                    feats_rst = self.renorm4t2m(feats_rst)
                    motions = self.renorm4t2m(motions)

                    # t2m motion encoder
                    m_lens = lengths.copy()
                    m_lens = torch.tensor(m_lens, device=motions.device)
                    align_idx = np.argsort(m_lens.data.tolist())[::-1].copy()
                    motions = motions[align_idx]
                    m_lens = m_lens[align_idx]
                    m_lens = torch.div(m_lens, self.cfg.data.unit_length, rounding_mode="floor")

                    recons_mov = self.t2m_moveencoder(feats_rst[..., :-4]).detach()
                    recons_emb = self.t2m_motionencoder(recons_mov, m_lens)
                    motion_mov = self.t2m_moveencoder(motions[..., :-4]).detach()
                    motion_emb = self.t2m_motionencoder(motion_mov, m_lens)

                    # t2m text encoder
                    text_emb = self.t2m_textencoder(word_embs, pos_ohot, text_lengths)[align_idx]

                    self.TM2TMetrics.update(
                        text_embeddings=text_emb,
                        recmotion_embeddings=recons_emb,
                        gtmotion_embeddings=motion_emb,
                        lengths=batch["length"]
                    )

                    loop.set_description(f'Test  Epoch [{epoch+1}/{self.cfg.train.epochs}]')

                result = self.TM2TMetrics.compute()
                FID_list.append(result['FID'])
        elif self.condition == 'action':
            mae = model['mae']
            denoiser_ema = model['denoiser_ema']
            condition_encoder = model['condition']
            mae.eval()
            denoiser_ema.eval()
            condition_encoder.eval()

            FID_list = []
            for repeat_idx in range(1):
                self.HUMANACTMetrics.init()

                loop = tqdm(enumerate(test_dataloader), total = len(test_dataloader))
                for idx, batch in loop:
                    actions = batch["action"]
                    motions = batch["motion"].detach().clone()
                    lengths = batch["length"]

                    feats_rst = diffusion.sample(bs=len(lengths), lengths=lengths, condition=actions, steps=self.cfg.train.sample_steps, denoiser=denoiser_ema, motion_encoder=mae,
                                                condition_encoder=condition_encoder, device=self.device)
                    
                    mask = batch["mask"]
                    joints_rst = self.feats2joints(feats_rst.cpu(), mask)
                    joints_ref = self.feats2joints(motions.cpu(), mask)
                    joints_eval_rst = self.feats2joints_eval(feats_rst.cpu(), mask)
                    joints_eval_ref = self.feats2joints_eval(motions.cpu(), mask)

                    rs_set = {
                        "m_action": actions,
                        "m_lens": lengths,
                        "joints_rst": joints_rst,
                        "joints_ref": joints_ref,
                        "joints_eval_rst": joints_eval_rst,
                        "joints_eval_ref": joints_eval_ref,
                    }

                    self.HUMANACTMetrics.update(rs_set["m_action"],
                                                rs_set["joints_eval_rst"],
                                                rs_set["joints_eval_ref"],
                                                rs_set["m_lens"])
                    
                result = self.HUMANACTMetrics.compute()
                print(result)
                FID_list.append(result['FID'])
        else:
            assert 0, condition_error

        return FID_list
    