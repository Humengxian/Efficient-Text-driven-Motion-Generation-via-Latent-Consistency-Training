import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

def _get_clone(module):
    return copy.deepcopy(module)

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Backbone(nn.Module):
    def __init__(self, seq_max, token_num, in_feat, dim, ff_dim, layer, dropout, timestep_condition=False):
        super().__init__()

        self.dim = dim

        # feats embedding
        self.feat_embeeding = nn.Linear(in_feat, dim)

        # position embedding
        self.position_embedding = nn.Parameter(torch.randn(1, seq_max + token_num, dim))
        self.timestep_condition = timestep_condition

        # backbone
        encode_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=8,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )

        self.backbone = SkipEncodeTransformer(encode_layer=encode_layer, num_layers=layer, dim=dim, norm=nn.LayerNorm(dim))

        # feats token 
        self.feat_token = nn.Parameter(torch.randn(1, token_num, dim))
        self.feat_token_mask = torch.zeros((1, token_num)).bool()

    def encode(self, feat, src_key_padding_mask=None):
        bs, seq_len, in_feat = feat.shape

        # embedding
        feat = self.feat_embeeding(feat)

        # concate token
        feat = torch.cat((self.feat_token.repeat(bs, 1, 1), feat), dim=1)
        if self.feat_token_mask.device != feat.device:
            self.feat_token_mask = self.feat_token_mask.to(feat.device)

        src_key_padding_mask = torch.cat((self.feat_token_mask.repeat(bs, 1), src_key_padding_mask), dim=1)
        feat = feat + self.position_embedding

        # backbone
        feat = self.backbone(feat, src_key_padding_mask=src_key_padding_mask)
        return feat
    
    def decode(self, token, src_key_padding_mask=None):
        bs, token_num, _ = token.shape

        # concate token
        feat = self.position_embedding.repeat(bs, 1, 1)
        feat[:, :token_num] = feat[:, :token_num] + token

        if self.timestep_condition:
            if self.feat_token_mask.device != feat.device:
                self.feat_token_mask = self.feat_token_mask.to(feat.device)
            src_key_padding_mask = torch.cat((self.feat_token_mask.repeat(bs, 1), src_key_padding_mask), dim=1)
        else:
            src_key_padding_mask = None

        # backbone
        feat = self.backbone(feat)
        return feat

class SkipEncodeTransformer(nn.Module):
    def __init__(self, encode_layer, num_layers, dim, norm=None) -> None:
        super().__init__()

        assert num_layers % 2 == 1

        self.d_model = dim

        num_block = (num_layers - 1) // 2
        self.input_blocks = _get_clones(encode_layer, num_block)
        self.middle_block = _get_clone(encode_layer)
        self.output_blocks = _get_clones(encode_layer, num_block)
        self.linear_blocks = _get_clones(nn.Linear(2 * self.d_model, self.d_model), num_block)
        self.norm = norm
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, src_key_padding_mask=None):
        x = src

        xs = []
        for module in self.input_blocks:
            x = module(x, src_key_padding_mask=src_key_padding_mask)
            xs.append(x)

        x = self.middle_block(x, src_key_padding_mask=src_key_padding_mask)

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, src_key_padding_mask=src_key_padding_mask)

        if self.norm is not None:
            x = self.norm(x)
        return x


class SkipDecodeTransformer(nn.Module):
    def __init__(self, decode_layer, num_layers, dim, norm=None) -> None:
        super().__init__()

        assert num_layers % 2 == 1

        self.d_model = dim

        num_block = (num_layers - 1) // 2
        self.input_blocks = _get_clones(decode_layer, num_block)
        self.middle_block = _get_clone(decode_layer)
        self.output_blocks = _get_clones(decode_layer, num_block)
        self.linear_blocks = _get_clones(nn.Linear(2 * self.d_model, self.d_model), num_block)
        self.norm = norm
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, memory, src_key_padding_mask=None, memory_key_padding_mask=None):
        x = src

        xs = []
        for module in self.input_blocks:
            x = module(x, memory, tgt_key_padding_mask=src_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
            xs.append(x)

        x = self.middle_block(x, memory, tgt_key_padding_mask=src_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

        for (module, linear) in zip(self.output_blocks, self.linear_blocks):
            x = torch.cat([x, xs.pop()], dim=-1)
            x = linear(x)
            x = module(x, memory, tgt_key_padding_mask=src_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)

        if self.norm is not None:
            x = self.norm(x)
        return x
