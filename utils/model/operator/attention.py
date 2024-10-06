import torch
from torch import nn
import math

class Attention(nn.Module):
    def __init__(self, d_in, d_model: int, nhead: int, dropout: float) -> None:
        super().__init__()

        self.num_attention_heads = nhead
        self.attention_head_size = int(d_model / nhead)

        self.query = nn.Linear(d_in, d_model)
        self.key = nn.Linear(d_in, d_model)
        self.value = nn.Linear(d_in, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, encoder_hidden_states=None):
        Q = self.query(hidden_states)
        K = self.key(encoder_hidden_states)
        V = self.value(encoder_hidden_states)

        attention_scores = torch.matmul(Q, K.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, V)
        return context_layer

class AttentionOut(nn.Module):
    def __init__(self, d_in, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.dense_input = nn.Linear(d_in, d_model)
        # nn.init.constant_(self.dense.weight, 0)

    def forward(self, input_tensor, hidden_states):
        hidden_states = self.dense(hidden_states)
        return hidden_states + self.dense_input(input_tensor)

class MultiScaleMatchModule(nn.Module):
    def __init__(self, key_dim, lowdim) -> None:
        super().__init__()
        self.map = nn.Linear(key_dim, lowdim)

    def forward(self, c_text, g_text, g_motion):
        Q = c_text
        K = g_text
        V = g_motion

        attention_scores = torch.matmul(Q, K.transpose(-1, -2))

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        out = attention_probs[:, 0][:, :, None, None] * V
        out = torch.sum(out, dim=1, keepdim=False)
        
        return out