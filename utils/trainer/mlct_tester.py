from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import json
import torch
import torch.nn.functional as F
from .monitor import AverageMeter
from tqdm import tqdm
from ..data.utils import lengths_to_mask
from ..model.operator.clip_module import BertTextEncoder, CLIPTextEncoder
from ..model.diffusion import Diffusion
from ..model.network import t2m_motionenc, t2m_textenc
from ..visual.plot_script import plot_3d_motion
from ..visual import paramUtil 
import numpy as np
from ..metrics.tm2t import TM2TMetrics
from ..metrics.mm import MMMetrics
from rich.table import Table
from rich import get_console

def print_table(title, metrics):
    table = Table(title=title)

    table.add_column("Metrics", style="cyan", no_wrap=True)
    table.add_column("Value", style="magenta")

    for key, value in metrics.items():
        table.add_row(key, str(value))

    console = get_console()
    console.print(table, justify="center")

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

def get_metric_statistics(values, replication_times):
    mean = np.mean(values, axis=0)
    std = np.std(values, axis=0)
    conf_interval = 1.96 * std / np.sqrt(replication_times)
    return mean, conf_interval

class MLCTTester(object):
    def __init__(self, cfg, logger) -> None:
        self.cfg = cfg
        self.logger = logger
        self.feats2joints = None
        self.device = cfg.device + str(cfg.gpu)

        self.TM2TMetrics = TM2TMetrics(
            diversity_times=cfg.test.diversity_times
        )

        self.MMMetrics = MMMetrics(
            mm_num_times=cfg.test.mm_num_times
        )
        self._get_t2m_evaluator(cfg)

    def test(self, model, datamodule):
        cfg = self.cfg
        self.skeleton = paramUtil.kit_kinematic_chain if cfg.data.name == 'kit' else paramUtil.t2m_kinematic_chain

        # prepare model
        mae = model['mae']
        denoiser = model['denoiser']
        if 'clip' in cfg.diffusion.text_path or "clip_path" in cfg.diffusion.keys():
            model['condition'] = CLIPTextEncoder(cfg.diffusion.clip_path if "clip_path" in cfg.diffusion.keys() else cfg.diffusion.text_path, last_hidden_state=False).to(self.device)
        else:
            model['condition'] = BertTextEncoder(cfg.diffusion.text_path, last_hidden_state=False).to(self.device)
        mae = mae.to(self.device)
        denoiser = denoiser.to(self.device)
        
        # prepare
        diffusion = Diffusion(cfg)

        # recover joints function
        if cfg.condition == 'text':
            self.feats2joints = datamodule.feats2joints
            self.renorm4t2m = datamodule.renorm4t2m

        # dataloader
        test_dataloader = datamodule.test_dataloader()

        all_metrics = {}
        for epoch in range(cfg.test.replication_times):
            wo_mm_metrics = self.test_wo_mm_step(model, test_dataloader, diffusion, epoch)
            for key, item in wo_mm_metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = [item]
                else:
                    all_metrics[key] += [item]

            name_list = test_dataloader.dataset.name_list
            mm_list = np.random.choice(name_list, cfg.test.mm_num_samples, replace=False)
            test_dataloader.dataset.name_list = mm_list

            mm_metrics = self.test_mm_step(model, test_dataloader, diffusion, epoch)
            for key, item in mm_metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = [item]
                else:
                    all_metrics[key] += [item]

            test_dataloader.dataset.name_list = name_list

            wo_mm_metrics.update(mm_metrics)
            self.logger.info(str(wo_mm_metrics))

        all_metrics_new = {}
        for key, item in all_metrics.items():
            mean, conf_interval = get_metric_statistics(np.array(item), cfg.test.replication_times)
            all_metrics_new[key + "/mean"] = mean
            all_metrics_new[key + "/conf_interval"] = conf_interval
        print_table(f"Mean Metrics", all_metrics_new)

        for key, item in all_metrics_new.items():
            all_metrics_new[key] = str(item)

        with open(os.path.join(cfg.output_dir, 'result.json'), "w", encoding="utf-8") as f:
            json.dump(all_metrics_new, f, indent=4)

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
        t2m_checkpoint = torch.load(os.path.join('deps', dataname, "text_mot_match/model/finest.tar"), map_location="cuda:0")
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
    def test_wo_mm_step(self, model, test_dataloader, diffusion, epoch):
        mae = model['mae']
        denoiser = model['denoiser']
        condition_encoder = model['condition']
        mae.eval()
        denoiser.eval()
        condition_encoder.eval()
        
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

            feats_rst = diffusion.sample(bs=len(lengths), lengths=lengths, condition=texts, steps=self.cfg.train.sample_steps, denoiser=denoiser, motion_encoder=mae,
                                        condition_encoder=condition_encoder, device=self.device)

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

            loop.set_description(f'Test  Epoch [{epoch+1}/{self.cfg.test.replication_times}]')

        result = self.TM2TMetrics.compute()

        return result
    
    @torch.no_grad()
    def test_mm_step(self, model, test_dataloader, diffusion, epoch):
        mae = model['mae']
        denoiser = model['denoiser']
        condition_encoder = model['condition']
        mae.eval()
        denoiser.eval()
        condition_encoder.eval()

        self.MMMetrics.init()

        loop = tqdm(enumerate(test_dataloader), total = len(test_dataloader))
        for idx, batch in loop:
            texts = batch["text"]
            motions = batch["motion"].detach().clone()
            lengths = batch["length"]
            word_embs = batch["word_embs"].detach().clone()
            pos_ohot = batch["pos_ohot"].detach().clone()
            text_lengths = batch["text_len"].detach().clone()

            texts = texts * self.cfg.test.mm_num_repeats
            motions = motions.repeat_interleave(self.cfg.test.mm_num_repeats, dim=0)
            lengths = lengths * self.cfg.test.mm_num_repeats
            word_embs = word_embs.repeat_interleave(self.cfg.test.mm_num_repeats, dim=0)
            pos_ohot = pos_ohot.repeat_interleave(self.cfg.test.mm_num_repeats, dim=0)
            text_lengths = text_lengths.repeat_interleave(self.cfg.test.mm_num_repeats, dim=0)

            motions = motions.to(self.device)
            word_embs = word_embs.to(self.device)
            pos_ohot = pos_ohot.to(self.device)
            text_lengths = text_lengths.to(self.device)

            feats_rst = diffusion.sample(bs=len(lengths), lengths=lengths, condition=texts, steps=self.cfg.train.sample_steps, denoiser=denoiser, motion_encoder=mae,
                                        condition_encoder=condition_encoder, device=self.device)

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

            self.MMMetrics.update(
                mm_motion_embeddings=recons_emb.unsqueeze(0),
                lengths=batch["length"]
            )

            loop.set_description(f'Test  Epoch [{epoch+1}/{self.cfg.test.replication_times}]')

        result = self.MMMetrics.compute()

        return result
