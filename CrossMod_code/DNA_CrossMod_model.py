# -*- coding: utf-8 -*-
import torch
from torch import nn
import math


# ---------------------------
# Shortcut MLP
# ---------------------------
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


# ============================================================
# Cross-Attention (No RoPE)
# ============================================================
class BiCrossAttention(nn.Module):
    def __init__(self, dim, nhead):
        super().__init__()

        self.dim = dim
        self.nhead = nhead
        self.head_dim = dim // nhead

        # ESM -> Saprot
        self.q1 = nn.Linear(dim, dim)
        self.k1 = nn.Linear(dim, dim)
        self.v1 = nn.Linear(dim, dim)

        # Saprot -> ESM
        self.q2 = nn.Linear(dim, dim)
        self.k2 = nn.Linear(dim, dim)
        self.v2 = nn.Linear(dim, dim)

        # Output projection
        self.out = nn.Linear(dim * 2, dim)

        self.dropout = nn.Dropout(0.1)

    def forward(self, esm, saprot):
        B, L, D = esm.shape
        H = self.nhead

        # -------------------------------------------------
        # ESM Query ; Saprot Key + Value
        # -------------------------------------------------
        q1 = self.q1(esm).view(B, L, H, self.head_dim).transpose(1, 2)
        k1 = self.k1(saprot).view(B, L, H, self.head_dim).transpose(1, 2)
        v1 = self.v1(saprot).view(B, L, H, self.head_dim).transpose(1, 2)

        score1 = torch.einsum("bhld,bhmd->bhlm", q1, k1) / math.sqrt(self.head_dim)
        a1 = torch.softmax(score1, dim=-1)

        a1 = torch.einsum("bhlm,bhmd->bhld", a1, v1)
        a1 = a1.transpose(1, 2).reshape(B, L, D)

        # -------------------------------------------------
        # Saprot Query ; ESM Key + Value
        # -------------------------------------------------
        q2 = self.q2(saprot).view(B, L, H, self.head_dim).transpose(1, 2)
        k2 = self.k2(esm).view(B, L, H, self.head_dim).transpose(1, 2)
        v2 = self.v2(esm).view(B, L, H, self.head_dim).transpose(1, 2)

        score2 = torch.einsum("bhld,bhmd->bhlm", q2, k2) / math.sqrt(self.head_dim)
        a2 = torch.softmax(score2, dim=-1)

        a2 = torch.einsum("bhlm,bhmd->bhld", a2, v2)
        a2 = a2.transpose(1, 2).reshape(B, L, D)

        # -------------------------------------------------
        # Concatenate
        # -------------------------------------------------
        out = torch.cat([a1, a2], dim=-1)  # (B, L, 2D)
        out = self.out(out)                # (B, L, D)

        return out


# ============================================================
# Cross-Attention Model (No RoPE + proj_dim=128)
# ============================================================
class TransformerWithBiCross(nn.Module):
    def __init__(
        self,
        esm_dim=1280,
        saprot_dim=1280,
        proj_dim=128,         
        nhead=8,
        dim_feedforward=512  
    ):
        super().__init__()

        # Feature projection
        self.proj_esm = nn.Linear(esm_dim, proj_dim)
        self.proj_sp = nn.Linear(saprot_dim, proj_dim)

        # Bi-directional Cross Attention
        self.cross = BiCrossAttention(proj_dim, nhead)

        # Transformer block
        self.norm1 = nn.LayerNorm(proj_dim)

        self.ffn = nn.Sequential(
            nn.Linear(proj_dim, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, proj_dim)
        )

        self.norm2 = nn.LayerNorm(proj_dim)

        # Prediction head
        self.PB_linear = LinearwithShortcut(
            in_features=proj_dim,
            hidden_features=128,
            out_features=1
        )

    def forward(self, esm_feature, saprot_feature):

        # Projection
        esm = self.proj_esm(esm_feature)
        sap = self.proj_sp(saprot_feature)

        # Cross Attention
        attn_out = self.cross(esm, sap)

        # Residual + Norm
        x = self.norm1(esm + attn_out)

        # FFN
        ffn_out = self.ffn(x)

        # Residual + Norm
        x = self.norm2(x + ffn_out)

        # Prediction
        y = self.PB_linear(x)

        return y