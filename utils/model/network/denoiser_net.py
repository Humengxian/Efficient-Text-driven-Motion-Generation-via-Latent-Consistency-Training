import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ..operator.attention import MultiScaleMatchModule
from ..operator.embedding import Timesteps
from ..operator.block import _get_clones, _get_clone

# From LCMScheduler.get_scalings_for_boundary_condition_discrete
def scalings_for_boundary_conditions(timestep, sigma_data=0.5, timestep_scaling=10.0):
    c_skip = sigma_data**2 / ((timestep / 0.1) ** 2 + sigma_data**2)
    c_out = (timestep / 0.1) / ((timestep / 0.1) ** 2 + sigma_data**2) ** 0.5
    return c_skip, c_out

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

class TransformerMotionDenoiser(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()

        latent_dim = cfg.motion_ae.dim
        diffuion_dim = cfg.diffusion.dim
        self.cfg = cfg
        self.node = cfg.motion_ae.node
        # self.sigmoid = sigmoid
        self.levels = torch.tensor([int(cfg.motion_ae.node) - 1 for _ in range(cfg.motion_ae.token_num)], dtype=torch.int32)
        
        self.latent_dim = latent_dim
        self.diffuion_dim = diffuion_dim
        self.token_num = cfg.motion_ae.token_num

        self.total_n_layer = cfg.diffusion.layer

        # position embedding
        self.position_emb = nn.Parameter(torch.randn((1, len(self.levels) + 2, diffuion_dim)))

        # feat embedding
        self.feat_embedding = nn.Linear(latent_dim, diffuion_dim)

        # text embedding
        text_embedding = nn.Sequential(
                nn.Linear(768, diffuion_dim),
                nn.GELU(),
                nn.Linear(diffuion_dim, diffuion_dim)
            )
        self.text_embedding = _get_clones(text_embedding, self.total_n_layer)

        # time embedding
        self.timesteps_map = Timesteps(latent_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        timesteps_embedding = nn.Sequential(
                nn.Linear(latent_dim, diffuion_dim),
                nn.GELU(),
                nn.Linear(diffuion_dim, diffuion_dim)
            )
        self.timesteps_embedding = _get_clones(timesteps_embedding, self.total_n_layer)
        
        # backbone
        backbone = nn.TransformerEncoderLayer(
                d_model=diffuion_dim,
                nhead=8,
                dim_feedforward=cfg.diffusion.ff_dim,
                dropout=cfg.diffusion.dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
        self.backbone = _get_clones(backbone, self.total_n_layer)

        # matching network
        self.match_net = MultiScaleMatchModule(768, cfg.diffusion.lowdim)
        match = nn.Linear(diffuion_dim, diffuion_dim)      
        self.match = _get_clones(match, self.total_n_layer)    
        for layer_ in self.match:
            nn.init.constant_(layer_.weight, 1e-8)                                                                                     

        # fuse
        fuse_layer = nn.Sequential(
            nn.LayerNorm(self.diffuion_dim * 2),
            nn.Linear(2 * self.diffuion_dim, self.diffuion_dim),
            nn.GELU(),
            nn.Linear(self.diffuion_dim, self.diffuion_dim)
        )
        self.fuse_layer = _get_clones(fuse_layer, self.total_n_layer // 2)

        self.motion_id_head_regressor = nn.Sequential(
                nn.Linear(diffuion_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim)
            )
        self.cluster_motion = None
        self.cluster_text = None

    def load_cluster_dict(self, cluster_dict):
        self.cluster_motion = torch.from_numpy(cluster_dict['motion']).to(self.cfg.device + str(self.cfg.gpu))[None, :, :, :].float()
        self.cluster_text = torch.from_numpy(cluster_dict['text']).to(self.cfg.device + str(self.cfg.gpu))[None, :, :].float()

    def forward(self, x_t, timesteps, text_emb):
        bs, d, latent_dim = x_t.shape
        # timestep embeeding
        time_emb = self.timesteps_map(timesteps).unsqueeze(1)

        # feat embedding
        feat_emb = self.feat_embedding(x_t)
        
        # concate
        x = torch.cat((torch.zeros((bs, 2, self.diffuion_dim), device=x_t.device), feat_emb), dim=1) + self.position_emb
        
        # match
        if self.cluster_text is not None:
            refer_emb = self.match_net(text_emb, self.cluster_text, self.cluster_motion)
            refer_emb = self.feat_embedding(refer_emb)

        # backbone
        xs = []
        fuse_id = 0
        # print(refer_emb.shape)
        for idx, (backbone, text_embedding, timesteps_embedding, match_weight) in enumerate(zip(self.backbone, self.text_embedding, self.timesteps_embedding, self.match)):
            if idx >= (self.total_n_layer - self.total_n_layer // 2):
                xs_pop = xs.pop()
                x = self.fuse_layer[fuse_id](torch.cat([x, xs_pop], dim=-1))
                fuse_id += 1
            
            x[:, :1] = timesteps_embedding(time_emb)
            x[:, 1:2] = text_embedding(text_emb)
            if self.cluster_text is not None:
                x[:, 2:] = x[:, 2:] + match_weight(refer_emb)

            x = backbone(x)

            if idx < self.total_n_layer // 2:
                xs.append(x)

        # head
        out = self.motion_id_head_regressor(x[:, 2:])

        # out = out + refer

        c_skip, c_out = scalings_for_boundary_conditions(timesteps)
        c_skip, c_out = [append_dims(x, out.ndim) for x in [c_skip, c_out]]
        return c_skip * x_t + c_out * out
