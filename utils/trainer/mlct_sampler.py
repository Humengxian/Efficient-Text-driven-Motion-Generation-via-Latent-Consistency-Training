import time
import numpy as np
import os
import torch
from ..data.humanml.scripts.motion_process import (process_file, recover_from_ric)
from ..visual import paramUtil
from ..visual.plot_script import plot_3d_motion
from ..model.operator.clip_module import CLIPTextEncoder
from ..model.diffusion import Diffusion

def load_state_dict(path, model):
    checkpoint = torch.load(path)
    model_dict =  model.state_dict()
    state_dict = {k:v for k, v in checkpoint.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    return model

class Sampler(object):
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        # diffusion
        self.diffusion = Diffusion(cfg)

        # text encoder
        self.text_encoder = CLIPTextEncoder(cfg.diffusion.clip_path, last_hidden_state=False)
        self.text_encoder.eval().to(device=cfg.device + str(cfg.gpu))
        self.device = cfg.device + str(cfg.gpu)

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
        for i in range(50):
            sample = self.diffusion.sample(bs=len(lengths), lengths=lengths, condition=text, steps=step, denoiser=model['denoiser'], motion_encoder=model['mae'],
                                            condition_encoder=self.text_encoder, device=self.device)
            
        start_time = time.time()
        for i in range(100):
            sample = self.diffusion.sample(bs=len(lengths), lengths=lengths, condition=text, steps=step, denoiser=model['denoiser'], motion_encoder=model['mae'],
                                                condition_encoder=self.text_encoder, device=self.device)
        print((time.time() - start_time) / 100)

        # recover
        joints = self.feats2joints(sample)
        if bs == 1:
            joints = joints[0, :lengths[0]].detach().cpu().numpy()
            # joints = joints[len_mask].detach().cpu().numpy()
            plot_3d_motion(os.path.join(self.cfg.sample_dir, f"{file_name}.gif"), self.skeleton, joints, dataset=self.cfg.data.name, title=text[0], fps=self.fps)
            np.save(os.path.join(self.cfg.sample_dir, f"{file_name}.npy"), joints)
        else:
            for _bs in range(bs):
                # joint = joints[_bs, len_mask[_bs]].detach().cpu().numpy()
                joint = joints[_bs, :lengths[_bs]].detach().cpu().numpy()
                plot_3d_motion(os.path.join(self.cfg.sample_dir, f"{file_name}_{_bs}.gif"), self.skeleton, joint, dataset=self.cfg.data.name, title=text[_bs], fps=self.fps)
        
                np.save(os.path.join(self.cfg.sample_dir, f"{file_name}_{_bs}.npy"), joint)

    def sample_gt(self, file, id_, text):
        joints = recover_from_ric(torch.from_numpy(np.float32(file)).unsqueeze(0), self.cfg.data.joints_num)
        joints = joints.detach().cpu().numpy()[0]
        plot_3d_motion(os.path.join(self.cfg.sample_dir, f"{id_}_GT.gif"), self.skeleton, joints, dataset=self.cfg.data.name, title=text, fps=self.fps)
        np.save(os.path.join(self.cfg.sample_dir, f"{id_}_GT.npy"), joints)