import torch
import torch.nn as nn
import torch.nn.functional as F
from ..operator.block import Backbone

class TransformerMotionAutoEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        # para
        self.cfg = cfg
        self.dim = cfg.motion_ae.dim

        self.type_ = cfg.motion_ae.type
        assert self.type_ in ['qae', 'vae', 'ae']
        self.levels = torch.tensor([int(cfg.motion_ae.node) for _ in range(cfg.motion_ae.token_num)], dtype=torch.int32)
        self.token_num = cfg.motion_ae.token_num

        # motion encoder
        self.motion_encoder = Backbone(seq_max=cfg.data.max_motion_length,
                                       token_num=self.token_num if (self.type_ == 'qae' or self.type_ == 'ae') else self.token_num * 2,
                                       in_feat=cfg.data.dim_pose,
                                       dim=cfg.motion_ae.dim,
                                       ff_dim=cfg.motion_ae.ff_dim,
                                       layer=cfg.motion_ae.encode_layer,
                                       dropout=cfg.motion_ae.dropout)
        
        # motion decoder 
        self.motion_decoder = Backbone(seq_max=cfg.data.max_motion_length,
                                       token_num=len(self.levels),
                                       in_feat=cfg.data.dim_pose,
                                       dim=cfg.motion_ae.dim,
                                       ff_dim=cfg.motion_ae.ff_dim,
                                       layer=cfg.motion_ae.decode_layer,
                                       dropout=cfg.motion_ae.dropout,
                                       timestep_condition=True)

        # motion head
        self.motion_recon_head = nn.Sequential(
            nn.Linear(self.dim, cfg.data.dim_pose),
            nn.ReLU(),
            nn.Linear(cfg.data.dim_pose, cfg.data.dim_pose)
        )

    def quantized(self, token, eps=1e-6):
        if self.levels.device != token.device:
            self.levels = self.levels.to(token.device)

        l = self.levels.unsqueeze(0).unsqueeze(-1)
        token = token.tanh() * l
        token_hat = token.round()
        quantized_token_hat =  token + (token_hat - token).detach()
        quantized_token = quantized_token_hat / l

        return quantized_token, token_hat
        
    def motion_encode_decode(self, motion, mask):
        bs, seq_len, d = motion.shape
        
        # encode
        if self.type_ == 'qae' or self.type_ == "ae":
            latent, _ = self.motion_encode(motion, mask)
        else:
            latent, dist_pred = self.motion_encode(motion, mask)
        
        # decode
        pred_motion = self.motion_decode(latent, mask)

        if self.type_ == 'qae' or self.type_ == "ae":
            return pred_motion
        else:
            return pred_motion, dist_pred

    def motion_encode(self, motion, mask):
        bs, seq_len, d = motion.shape

        # encode
        token_num = self.token_num if (self.type_ == 'qae' or self.type_ == 'ae') else self.token_num * 2
        motion_feat = self.motion_encoder.encode(motion, src_key_padding_mask=mask)[:, :token_num]
        
        if self.type_ == 'qae':
            # quantized
            latent, latent_id = self.quantized(motion_feat)
            return latent, latent_id
        elif self.type_ == 'ae':
            return motion_feat, None
        else:
            mu = motion_feat[:, :self.token_num]
            logvar = motion_feat[:, self.token_num:]
            std = logvar.exp().pow(0.5)

            dist = torch.distributions.Normal(mu, std)
            latent = dist.rsample()
            return latent, dist
        
    def motion_decode(self, latent, mask):
        # decode
        motion_feat = self.motion_decoder.decode(latent, src_key_padding_mask=mask)[:, len(self.levels):]
        
        # head
        pred_motion = self.motion_recon_head(motion_feat)

        return pred_motion

