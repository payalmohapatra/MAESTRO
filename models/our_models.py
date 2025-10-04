import os
import pickle
import torch
import numpy as np
import pandas as pd
import sys
import math
## General pytorch libraries
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from math import sqrt, ceil

# append the data 
sys.path.append('./data/')
sys.path.append('./utils/')
sys.path.append('./models/')

from utils import *


class ProbSparseAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1, factor=5, scale=None):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.factor = factor
        self.scale = scale or 1. / sqrt(self.d_head)

        self.q_linear = nn.Linear(d_model, d_model)
        self.kv_linear = nn.Linear(d_model, 2*d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, factor):
        B, L, _ = x.size()

        # Linear projection
        Q = self.q_linear(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)  # [B, H, L, D]
        K, V = self.kv_linear(x).chunk(2, dim=-1)
        K = K.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        U_part = min(factor * ceil(np.log(L)), L)
        u = min(factor * ceil(np.log(L)), L)

        # Sample QK for importance estimation
        index_sample = torch.randint(L, (L, U_part), device=x.device)
        K_sample = K.unsqueeze(2).expand(-1, -1, L, -1, -1)
        K_sample = K_sample[:, :, torch.arange(L).unsqueeze(1), index_sample, :]  # [B, H, L, U_part, D]
        QK_sampled = torch.matmul(Q.unsqueeze(3), K_sample.transpose(-2, -1)).squeeze(3)  # [B, H, L, U_part]
        M = QK_sampled.max(-1)[0] - QK_sampled.mean(-1)
        _, top_idx = torch.topk(M, u, dim=-1)  # [B, H, u]

        # Select top queries
        top_queries = Q.gather(2, top_idx.unsqueeze(-1).expand(-1, -1, -1, self.d_head))  # [B, H, u, D]
        scores = torch.matmul(top_queries, K.transpose(-2, -1)) * self.scale  # [B, H, u, L]
        attn = F.softmax(scores, dim=-1)
        context_update = torch.matmul(attn, V)  # [B, H, u, D]

        # Initialize context (mean over V)
        context = V.mean(dim=2, keepdim=True).expand(-1, -1, L, -1).clone()
        # Scatter updated top-u query contexts back
        batch_idx = torch.arange(B)[:, None, None]
        head_idx = torch.arange(self.n_heads)[None, :, None]
        context[batch_idx, head_idx, top_idx, :] = context_update

        # Merge heads
        context = context.transpose(1, 2).contiguous().view(B, L, self.d_model)
        return self.out(context)



class SparseMoEFeedForward(nn.Module):
    def __init__(self, d_model, expert_dim=256, num_experts=4, k=1, log_activations=True):
        super().__init__()
        self.num_experts = num_experts
        self.k = k
        self.log_activations = log_activations
        self.logged_expert_ids = []

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, expert_dim),
                nn.ReLU(),
                nn.Linear(expert_dim, d_model)
            ) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts)

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.reshape(B * T, D)  # Replace view with reshape

        gate_scores = self.gate(x_flat)  # [B*T, num_experts]
        topk_scores, topk_indices = torch.topk(gate_scores, self.k, dim=-1)

        if self.log_activations:
            self.logged_expert_ids.append(topk_indices.detach().cpu())

        output = torch.zeros_like(x_flat)
        for i in range(self.k):
            expert_ids = topk_indices[:, i]
            one_hot_mask = F.one_hot(expert_ids, self.num_experts).bool()
            for expert_idx in range(self.num_experts):
                expert_mask = one_hot_mask[:, expert_idx]
                if expert_mask.sum() == 0:
                    continue
                selected = x_flat[expert_mask]
                result = self.experts[expert_idx](selected)
                score = topk_scores[expert_mask, i].unsqueeze(1)
                output[expert_mask] += score * result

        return output.reshape(B, T, D)  # Use reshape here as well

    def get_activation_logs(self):
        # [steps, B*T, k]
        return torch.cat(self.logged_expert_ids, dim=0).numpy() if self.logged_expert_ids else None


class ModalityPositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len, num_modalities):
        super().__init__()
        self.temporal_pe = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.modality_pe = nn.Embedding(num_modalities, d_model)
        nn.init.normal_(self.temporal_pe, mean=0, std=0.02)

    def forward(self, x, modality_id):
        """
        x: [B, T, D] — modality input
        modality_id: int — modality index
        """
        B, T, D = x.shape
        pe = self.temporal_pe[:, :T, :]         # [1, T, D]
        me = self.modality_pe(torch.tensor(modality_id, device=x.device))  # [D]
        me = me.view(1, 1, D).expand(B, T, D)    # [B, T, D]
        return x + pe + me

class TemporalPositionalEncoder(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)  # [T, D]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [T, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))  # [D/2]
        
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        pe = pe.unsqueeze(0)  # [1, T, D]
        
        self.register_buffer('pe', pe)  

    def forward(self, x):
        """
        x: [B, T, D] — modality input
        """
        B, T, D = x.shape
        return x + self.pe[:, :T, :]


class InformerEncoderLayerWithMoE(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, factor=5, num_experts=4, k=1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        # Self-Attention (ProbSparseAttention)
        self.attention = ProbSparseAttention(d_model, n_heads, dropout, factor)
        
        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Distilling Convolution (downsampling)
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # SparseMoE Feed-Forward Layer
        self.moe_ff = SparseMoEFeedForward(d_model, d_model, num_experts=num_experts, k=k)

    def forward(self, x, factor):
        # Self-attention
        attn_out = self.attention(x, factor)
        x = self.norm1(x + attn_out)
        
        # Distilling (downsampling)
        x = x.permute(0, 2, 1)  # [batch, features, time]
        x = self.conv(x) + self.pool(x)
        x = x.permute(0, 2, 1)  # back to [batch, time, features]
        
        # SparseMoE Feed-forward
        ff_out = self.moe_ff(x)
        
        # Adding residual connections
        x = self.norm2(x + ff_out)
        
        return x
    

def modality_dropout(x, modalities, variates, dropout_prob=0.2, training=True):
    # x: [B, T, total_features]
    # Returns x with some modalities zeroed out and a mask
    B, T, _ = x.shape
    modality_mask = []
    start_idx = 0
    x_dropped = x.clone()
    for idx, modality in enumerate(modalities):
        num_vars = variates[modality]
        if training and torch.rand(1).item() < dropout_prob:
            x_dropped[:, :, start_idx:start_idx+num_vars] = 0
            modality_mask.append(0)
        else:
            modality_mask.append(1)
        start_idx += num_vars
    modality_mask = torch.tensor(modality_mask, device=x.device).float()  # [num_modalities]
    return x_dropped, modality_mask




# Refactored from original informer Encoder : https://github.com/zhouhaoyi/Informer2020/blob/main/models/encoder.py
class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1, factor=5):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        self.attention = ProbSparseAttention(d_model, n_heads, dropout, factor)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Distilling convolution
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, stride=2)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Feed forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, factor):
        # Self-attention
        attn_out = self.attention(x, factor)
        x = self.norm1(x + attn_out)
        
        # Distilling (downsampling)
        x = x.permute(0, 2, 1)  # [batch, features, time]
        x = self.conv(x) + self.pool(x)
        x = x.permute(0, 2, 1)  # back to [batch, time, features]
        
        # Feed forward
        ff_out = self.ff(x)
        return self.norm2(x + ff_out)

class InformerClf(nn.Module):
    def __init__(self, input_channels, num_classes, input_length=256,
                 d_model=64, nhead=8, num_layers=2, dropout=0.1, factor=5):
        super().__init__()
        
        self.input_projection = nn.Linear(input_channels, d_model)
        self.pos_encoder = nn.Parameter(torch.zeros(1, input_length, d_model))
        nn.init.normal_(self.pos_encoder, mean=0, std=0.02)
        
        # Informer encoder
        self.encoder = nn.ModuleList([
            InformerEncoderLayer(
                d_model=d_model,
                n_heads=nhead,
                d_ff=d_model*4,
                dropout=dropout,
                factor=factor
            ) for _ in range(num_layers)
        ])
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # Input shape: [batch, time, features]
        x = self.input_projection(x) + self.pos_encoder
        
        for layer in self.encoder:
            x = layer(x)
            
        # Final global average pooling
        x = torch.mean(x, dim=1)
        return self.classifier(x)



class CrossAttnTransformerClf(nn.Module):
    def __init__(self, cfg, num_classes, input_length=256, d_model=64, nhead=8, num_layers_per_modal=2, num_layers=2, dropout=0.1, verbose=True, base_factor=5, num_experts=4):
        super().__init__()
        self.modalities = cfg.modalities
        self.variates = cfg.variates
        self.num_modalities = len(self.modalities)
        self.input_length = input_length
        self.verbose = verbose
        self.d_model = d_model
        self.base_factor = base_factor
        self.num_experts = num_experts

        self.input_projections = nn.ModuleDict({
            modality: nn.Linear(self.variates[modality], d_model)
            for modality in self.modalities
        })

        # Positional encoder shared across modalities
        self.pos_encoder = ModalityPositionalEncoder(
            d_model=d_model,
            max_len=input_length,
            num_modalities=self.num_modalities
        )
        
        self.temporal_pos_encoder = TemporalPositionalEncoder(
            d_model=d_model,
            max_len=input_length
        )

        # Per-modality Informer layers
        self.per_modal_informers = nn.ModuleDict({
            modality: nn.ModuleList([
                InformerEncoderLayer(
                    d_model=d_model,
                    n_heads=nhead,
                    d_ff=d_model * 4,
                    dropout=dropout,
                    factor=5,
                ) for _ in range(num_layers_per_modal)
            ]) for modality in self.modalities
        })

        # Final fusion Informer with sparse MoE
        self.informer_encoder = nn.ModuleList([
            InformerEncoderLayerWithMoE(
                d_model=d_model,
                n_heads=nhead,
                d_ff=d_model * 4,
                dropout=dropout,
                factor=5,
                num_experts=num_experts,
                k=1
            ) for _ in range(num_layers)
        ])

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes)
        )

        self.factor_gate = nn.Sequential(
            nn.Linear(self.num_modalities, self.num_modalities),
            nn.ReLU(),
            nn.Linear(self.num_modalities, self.num_modalities),
            nn.Sigmoid()
      )

    def forward(self, x, modality_dropout_prob=0.2, training=True):
        """
        x: [B, T, total_features]
        Assumes modality-wise features are concatenated in order defined by self.modalities
        """

        projected_modalities = []
        start_idx = 0

        x, modality_mask = modality_dropout(x, self.modalities, self.variates, dropout_prob=modality_dropout_prob, training=training
        )
        dynamic_factor = self.factor_gate(modality_mask) * self.base_factor
        for idx, modality in enumerate(self.modalities):
            num_vars = self.variates[modality]
            x_m = x[:, :, start_idx:start_idx + num_vars]
            start_idx += num_vars
            # Projection
            x_m = self.input_projections[modality](x_m)
            factor = ceil(dynamic_factor[idx] * modality_mask[idx] + 1e-3 * (1 - modality_mask[idx]))

            # Add modality + temporal position encoding
            x_m = self.temporal_pos_encoder(x_m)

            # Pass through per-modality Informer layers
            for layer in self.per_modal_informers[modality]:
                x_m = layer(x_m, factor=factor)
            
            # After per-modal Add modality + temporal position encoding Again
            x_m = self.pos_encoder(x_m, modality_id=idx)
            
            projected_modalities.append(x_m)
            

        # Concatenate across modalities
        x_cat = torch.cat(projected_modalities, dim=1)  # [B, T_total, d_model]

        # Final Informer encoder with MoE
        for layer in self.informer_encoder:
            x_cat = layer(x_cat, self.base_factor)

        # Global average pooling
        x_pooled = torch.mean(x_cat, dim=1)

        return self.classifier(x_pooled), dynamic_factor