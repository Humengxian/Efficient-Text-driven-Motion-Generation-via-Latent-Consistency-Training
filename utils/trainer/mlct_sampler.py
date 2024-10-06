import time
import numpy as np
import os
import torch
from ..data.humanml.scripts.motion_process import (process_file, recover_from_ric)
from ..visual import paramUtil
from ..visual.plot_script import plot_3d_motion
from ..model.operator.clip_module import CLIPTextEncoder, BertTextEncoder
from ..model.diffusion import Diffusion
from scipy.ndimage import gaussian_filter

def load_state_dict(path, model):
    checkpoint = torch.load(path)
    model_dict =  model.state_dict()
    state_dict = {k:v for k, v in checkpoint.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    return model

def motion_temporal_filter(motion, sigma=1):
    motion = motion.reshape(motion.shape[0], -1)
    for i in range(motion.shape[1]):
        motion[:, i] = gaussian_filter(motion[:, i], sigma=sigma, mode="nearest")
    return motion.reshape(motion.shape[0], -1, 3)

class Sampler(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.device = cfg.device + str(cfg.gpu)

        # diffusion
        self.diffusion = Diffusion(cfg)

        # text encoder
        if 'clip' in cfg.diffusion.text_path:
            self.text_encoder = CLIPTextEncoder(cfg.diffusion.text_path, last_hidden_state=False).to(self.device)
            self.text_name = 'clip'
        else:
            self.text_encoder = BertTextEncoder(cfg.diffusion.text_path, last_hidden_state=False).to(self.device)
            self.text_name = 't5'

        self.text_encoder.eval()
        
        # mean & std
        data_root = cfg.data.root
        self.mean = np.load(os.path.join(data_root, "Mean.npy"))
        self.std = np.load(os.path.join(data_root, "Std.npy"))

        # para
        self.fps = 12.5 if self.cfg.data.name == 'kit' else 20
        self.skeleton = paramUtil.kit_kinematic_chain if cfg.data.name == 'kit' else paramUtil.t2m_kinematic_chain


    def feats2joints(self, features):
        mean = torch.tensor(self.mean).to(features)
        std = torch.tensor(self.std).to(features)
        features = features * std + mean
        return recover_from_ric(features, self.cfg.data.joints_num)

    @torch.no_grad()
    def forward(self, bs, text, length, model, step=5, file_name='sample'):
        lengths = torch.tensor([length for i in range(bs)]).long()

        # sample
        sample = self.diffusion.sample(bs=bs, 
                                       lengths=lengths, 
                                       condition=text, 
                                       steps=step, 
                                       denoiser=model['denoiser'], 
                                       motion_encoder=model['mae'],
                                       condition_encoder=self.text_encoder, 
                                       device=self.device)
        
        # recover
        joints = self.feats2joints(sample)
        
        if bs == 1:
            joints = joints[0, :lengths[0]].detach().cpu().numpy()
            joints = motion_temporal_filter(joints, sigma=2.5)
            # joints = joints[len_mask].detach().cpu().numpy()
            plot_3d_motion(os.path.join(self.cfg.sample_dir, f"{file_name}.gif"), self.skeleton, joints, dataset=self.cfg.data.name, title=text[0], fps=self.fps)
            np.save(os.path.join(self.cfg.sample_dir, f"{file_name}.npy"), joints)
        else:
            for _bs in range(bs):
                # joint = joints[_bs, len_mask[_bs]].detach().cpu().numpy()
                joint = joints[_bs, :lengths[_bs]].detach().cpu().numpy()
                joint = motion_temporal_filter(joint, sigma=2.5)
                plot_3d_motion(os.path.join(self.cfg.sample_dir, f"{file_name}_{_bs}.gif"), self.skeleton, joint, dataset=self.cfg.data.name, title=text[_bs], fps=self.fps)
        
                np.save(os.path.join(self.cfg.sample_dir, f"{file_name}_{_bs}.npy"), joint)

    def sample_gt(self, file, id_, text):
        joints = recover_from_ric(torch.from_numpy(np.float32(file)).unsqueeze(0), self.cfg.data.joints_num)
        joints = joints.detach().cpu().numpy()[0]
        plot_3d_motion(os.path.join(self.cfg.sample_dir, f"{id_}_GT.gif"), self.skeleton, joints, dataset=self.cfg.data.name, title=text, fps=self.fps)
        np.save(os.path.join(self.cfg.sample_dir, f"{id_}_GT.npy"), joints)