from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import torch
import torch.nn.functional as F
from .monitor import AverageMeter
from tqdm import tqdm
from ..data.utils import lengths_to_mask



class MAETrainer(object):
    def __init__(self, cfg, logger) -> None:
        self.cfg = cfg
        self.logger = logger
        self.feats2joints = None
        self.condition = cfg.condition
        self.device = cfg.device + str(cfg.gpu)

    def train(self, model, datamodule):
        mae = model['mae']
        mae = mae.to(self.device)
        cfg = self.cfg

        # optimizer
        optimizer = AdamW(lr=cfg.train.lr, params=mae.parameters())
        lr_scheduler = CosineAnnealingLR(optimizer, T_max=cfg.train.epochs, eta_min=cfg.train.lr_min, last_epoch=-1)

        # recover joints function
        if cfg.condition == 'text':
            self.feats2joints = datamodule.feats2joints

        # dataloader
        train_dataloader = datamodule.train_dataloader()
        test_dataloader = datamodule.test_dataloader()

        for epoch in range(cfg.train.epochs):
            # train step
            train_motion_loss, train_joint_loss = self.train_step(model, optimizer, train_dataloader, epoch=epoch)
            self.logger.info(f'Epoch {epoch} Train Motion Loss {train_motion_loss:.6f} Train Joint Loss {train_joint_loss:.6f}.')
            
            if (epoch + 1) % cfg.train.test_epochs == 0:
                test_motion_loss, test_joint_loss = self.test_step(model, test_dataloader, epoch=epoch)
                self.logger.info(f'Epoch {epoch} Test  Motion Loss {test_motion_loss:.6f} Test  Joint Loss {test_joint_loss:.6f}.')

            if (epoch + 1) % cfg.train.save_epochs == 0:
                torch.save(model['mae'].state_dict(), os.path.join(cfg.save_dir, f'Epoch_{epoch+1:04d}_recover_motion_loss_{test_motion_loss:.4f}_recover_joints_loss_{test_joint_loss:.4f}.pth'))

            lr_scheduler.step()

    def loss_fn(self, motion, pred_motion, mask, dist_m=None, condition='text'):
        if condition == 'action':
            loss = F.smooth_l1_loss(pred_motion[mask], motion[mask])
            return loss, loss, loss
        
        if self.cfg.train.loss == 'l1smooth':
            joints_pred = self.feats2joints(motion)
            joints_gt = self.feats2joints(pred_motion)
            recover_motion_loss = F.smooth_l1_loss(pred_motion[mask], motion[mask])
            recover_joints_loss = F.smooth_l1_loss(joints_pred[mask], joints_gt[mask])
            loss = recover_motion_loss + self.cfg.train.lambda_joint * recover_joints_loss

        if self.cfg.motion_ae.type == 'vae':
            mu_ref = torch.zeros_like(dist_m.loc)
            scale_ref = torch.ones_like(dist_m.scale)
            dist_ref = torch.distributions.Normal(mu_ref, scale_ref)
            kl_loss = (torch.distributions.kl_divergence(dist_m, dist_ref)).mean()
            loss = loss + self.cfg.train.lambda_kl * kl_loss

        return loss, recover_motion_loss, recover_joints_loss

    def train_step(self, model, optimizer, train_dataloader, epoch):
        mae = model['mae']
        recover_motion_loss_monitor = AverageMeter('recover_motion_loss')
        recover_joints_loss_monitor = AverageMeter('recover_joints_loss')

        mae.train()
        loop = tqdm((train_dataloader), total = len(train_dataloader))
        for batch in loop:
            optimizer.zero_grad()
            if self.condition == 'text':
                motion, text, length = batch['motion'], batch['text'], batch['length']
            elif self.condition == 'action':
                motion, action, length = batch['motion'], batch['action'], batch['length']
            else:
                assert 0, "condition must in ['text', 'action']"

            motion = motion.to(self.device)
            mask = lengths_to_mask(length, max_len=self.cfg.data.max_motion_length, device=self.device)
            
            if self.cfg.motion_ae.type == 'qae' or self.cfg.motion_ae.type == 'ae':
                motion_pred = mae.motion_encode_decode(motion, ~mask)
                dist_m = None
            else:
                motion_pred, dist_m = mae.motion_encode_decode(motion, ~mask)

            # calculate loss
            loss, recover_motion_loss, recover_joints_loss = self.loss_fn(motion, motion_pred, mask, dist_m, condition=self.condition)

            loss.backward()
            optimizer.step()
            recover_motion_loss_monitor.update(recover_motion_loss.item(), n=motion.shape[0])
            recover_joints_loss_monitor.update(recover_joints_loss.item(), n=motion.shape[0])

            loop.set_description(f'Train Epoch [{epoch+1}/{self.cfg.train.epochs}]')
            loop.set_postfix(motion = recover_motion_loss_monitor.avg, joint = recover_joints_loss_monitor.avg)
        
        return recover_motion_loss_monitor.avg, recover_joints_loss_monitor.avg
        
    def test_step(self, model, test_dataloader, epoch):
        mae = model['mae']
        recover_motion_loss_monitor = AverageMeter('recover_motion_loss')
        recover_joints_loss_monitor = AverageMeter('recover_joints_loss')

        mae.eval()
        loop = tqdm((test_dataloader), total = len(test_dataloader))
        with torch.no_grad():
            for batch in loop:
                if self.condition == 'text':
                    motion, text, length = batch['motion'], batch['text'], batch['length']
                elif self.condition == 'action':
                    motion, action, length = batch['motion'], batch['action'], batch['length']
                else:
                    assert 0, "condition must in ['text', 'action']"
                motion = motion.to(self.device)
                mask = lengths_to_mask(length, max_len=self.cfg.data.max_motion_length, device=self.device)
                
                if self.cfg.motion_ae.type == 'qae' or self.cfg.motion_ae.type == 'ae':
                    motion_pred = mae.motion_encode_decode(motion, ~mask)
                    dist_m = None
                else:
                    motion_pred, dist_m = mae.motion_encode_decode(motion, ~mask)

                # calculate loss
                loss, recover_motion_loss, recover_joints_loss = self.loss_fn(motion, motion_pred, mask, dist_m, condition=self.condition)

                recover_motion_loss_monitor.update(recover_motion_loss.item(), n=motion.shape[0])
                recover_joints_loss_monitor.update(recover_joints_loss.item(), n=motion.shape[0])

                loop.set_description(f'Test  Epoch [{epoch+1}/{self.cfg.train.epochs}]')
                loop.set_postfix(motion = recover_motion_loss_monitor.avg, joint = recover_joints_loss_monitor.avg)
        
        return recover_motion_loss_monitor.avg, recover_joints_loss_monitor.avg
    