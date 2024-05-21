import torch
import torch.nn as nn
import torch.nn.functional as F
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

        # position embedding
        self.position_emb = nn.Parameter(torch.randn((1, len(self.levels) + 1, diffuion_dim)))

        # feat embedding
        self.feat_embedding = nn.Linear(latent_dim, diffuion_dim)
        self.text_embedding = nn.ModuleList([
            nn.Linear(768, diffuion_dim)
            for _ in range(cfg.diffusion.layer)
        ])

        self.timesteps_map = Timesteps(latent_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.timesteps_embedding = nn.ModuleList([
            nn.Linear(latent_dim, diffuion_dim)
            for _ in range(cfg.diffusion.layer)
        ])
        
        encode_layer = nn.TransformerEncoderLayer(
                d_model=diffuion_dim,
                nhead=8,
                dim_feedforward=cfg.diffusion.ff_dim,
                dropout=cfg.diffusion.dropout,
                activation='gelu',
                batch_first=True
            )
        
        num_block = (cfg.diffusion.layer - 1) // 2
        self.input_blocks = _get_clones(encode_layer, num_block)
        self.middle_block = _get_clone(encode_layer)
        self.output_blocks = _get_clones(encode_layer, num_block)
        self.linear_blocks = _get_clones(nn.Linear(2 * self.diffuion_dim, self.diffuion_dim), num_block)
        self.norm = nn.LayerNorm(diffuion_dim)

        self.motion_id_head_regressor = nn.Sequential(
                nn.Linear(diffuion_dim, latent_dim),
                nn.GELU(),
                nn.Linear(latent_dim, latent_dim)
            )
        
        
    def forward(self, x_t, timesteps, latent_text):
        bs, d, latent_dim = x_t.shape
        # timestep embeeding
        time_emb = self.timesteps_map(timesteps).unsqueeze(1)

        # feat embedding
        feat_emb = self.feat_embedding(x_t)
        # print(feat_emb.shape)
        # concate
        feat = torch.cat((torch.zeros((bs, 1, self.diffuion_dim), device=x_t.device), feat_emb), dim=1) + self.position_emb
        
        # backbone
        x = feat
        _id = 0
        xs = []
        for module in self.input_blocks:
            # insert timesteps & text
            cond_latent = self.timesteps_embedding[_id](time_emb)
            cond_latent += self.text_embedding[_id](latent_text)

            x[:, :1] += cond_latent
            x = module(x)
            xs.append(x)
            _id += 1

        # insert timesteps & text
        cond_latent = self.timesteps_embedding[_id](time_emb)
        cond_latent += self.text_embedding[_id](latent_text)
        x[:, :1] += cond_latent
        x = self.middle_block(x)
        _id += 1

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            # insert timesteps & text
            cond_latent = self.timesteps_embedding[_id](time_emb)
            cond_latent += self.text_embedding[_id](latent_text)

            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)

            x[:, :1] += cond_latent
            x = module(x)
            _id += 1
            
        # head
        out = self.motion_id_head_regressor(x[:, 1:])

        c_skip, c_out = scalings_for_boundary_conditions(timesteps)
        c_skip, c_out = [append_dims(x, out.ndim) for x in [c_skip, c_out]]
        return c_skip * x_t + c_out * out