# -*- coding: utf-8 -*-
import torch
from torch import nn
import math

# --------------------------------------------------
# RoPE positional embedding
# --------------------------------------------------
def sinusoidal_position_embedding(batch_size, nums_head, max_len, output_dim, device):
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(-1)  # (max_len,1)
    ids = torch.arange(0, output_dim // 2, dtype=torch.float)
    theta = 10000 ** (-2 * ids / output_dim)
    embeddings = position * theta  # (max_len, output_dim//2)
    embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)  
    embeddings = embeddings.repeat((batch_size, nums_head, *([1] * len(embeddings.shape))))
    embeddings = embeddings.reshape(batch_size, nums_head, max_len, output_dim)
    return embeddings.to(device)

def RoPE(q, k):
    # q, k: (B, H, L, D)
    batch_size, nums_head, seq_len, head_dim = q.shape

    pos_emb = sinusoidal_position_embedding(batch_size, nums_head, seq_len, head_dim, q.device)

    cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)  # (B,H,L,D)
    sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

    # rotate half-dim
    q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.shape)
    k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape(k.shape)

    q = q * cos_pos + q2 * sin_pos
    k = k * cos_pos + k2 * sin_pos
    return q, k

# --------------------------------------------------
# Linear with Shortcut
# --------------------------------------------------
class LinearwithShortcut(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, activate_class=nn.ReLU):
        super().__init__()
        self.activate_function = activate_class()
        self.linear1 = nn.Linear(in_features, hidden_features)
        self.linear2 = nn.Sequential(
            self.activate_function,
            nn.Linear(hidden_features, hidden_features),
            self.activate_function,
            nn.Linear(hidden_features, hidden_features)
        )
        self.linear3 = nn.Sequential(
            self.activate_function,
            nn.Linear(hidden_features, hidden_features),
            self.activate_function,
            nn.Linear(hidden_features, hidden_features)
        )
        self.linear4 = nn.Sequential(
            self.activate_function,
            nn.Linear(hidden_features, out_features)
        )
        self.activate = nn.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = x + self.linear2(x)
        x = x + self.linear3(x)
        y = self.activate(self.linear4(x))
        return y

# --------------------------------------------------
# Transformer + RoPE (single input version)
# --------------------------------------------------
class TransformerWithROPE(nn.Module):

    def __init__(self, esm_dim=1280, nhead=8, dim_feedforward=1024):
        super().__init__()

        self.total_dim = esm_dim
        self.nhead = nhead

        if self.total_dim % self.nhead != 0:
            raise ValueError("total_dim must be divisible by nhead.")
        self.head_dim = self.total_dim // self.nhead

        # q/k/v projections
        self.q_proj = nn.Linear(self.total_dim, self.total_dim, bias=False)
        self.k_proj = nn.Linear(self.total_dim, self.total_dim, bias=False)
        self.v_proj = nn.Linear(self.total_dim, self.total_dim, bias=False)

        self.attn_dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(self.total_dim, self.total_dim)

        # feedforward
        self.ffn = nn.Sequential(
            nn.Linear(self.total_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, self.total_dim)
        )

        self.norm1 = nn.LayerNorm(self.total_dim)
        self.norm2 = nn.LayerNorm(self.total_dim)

        # output layer per residue
        self.PB_linear = LinearwithShortcut(self.total_dim, 256, 1)

    # -----------------------------
    # Multi-head attention with RoPE
    # -----------------------------
    def multihead_attention(self, x):
        B, L, C = x.shape
        H = self.nhead
        D = self.head_dim

        # project (B, H, L, D)
        q = self.q_proj(x).view(B, L, H, D).transpose(1, 2)
        k = self.k_proj(x).view(B, L, H, D).transpose(1, 2)
        v = self.v_proj(x).view(B, L, H, D).transpose(1, 2)

        # apply RoPE to ALL dimensions
        q, k = RoPE(q, k)

        scores = torch.einsum("bhld,bhmd->bhlm", q, k) / math.sqrt(D)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        attn_out = torch.einsum("bhlm,bhmd->bhld", attn_weights, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, L, C)

        return self.out_proj(attn_out)

    # -----------------------------
    # Forward
    # -----------------------------
    def forward(self, esm_feature):
        x = esm_feature  # ONLY one feature now

        attn_out = self.multihead_attention(x)
        x = self.norm1(x + attn_out)

        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return self.PB_linear(x)   # (B, L, 1)


# -------------------------
# quick test
# -------------------------
if __name__ == "__main__":
    B, L = 2, 16
    esm = torch.randn(B, L, 1280)
    model = TransformerWithROPE(esm_dim=1280, nhead=8)
    out = model(esm)
    print("out.shape:", out.shape)  # (B,L,1)
