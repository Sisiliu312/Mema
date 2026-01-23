import os
import numpy as np
import torch
import torch.nn as nn
import re
import torch.nn.functional as F
from datetime import datetime
from transformers.models.llama.modeling_llama import LlamaRMSNorm

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)
    
class SpatialGate(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, x, c_l):
        """
        Args:
            x: [B, N, D] - 原始特征
            c_l: [B, D] - context
        Returns:
            modulated: [B, N, D]
        """
        # 将context广播到每个位置
        c_expanded = c_l.unsqueeze(1).expand(-1, x.shape[1], -1)  # [B, N, D]
        
        # 基于context生成spatial-specific gate
        gate = self.conv(c_expanded + x)  # [B, N, D]
        print("gate max:", gate.max().item(), "min:", gate.min().item())
        num_total = gate.numel()
        num_zeros = (gate < 1e-3).sum().item()
        num_ones  = (gate > 1-1e-3).sum().item()
        print(f"gate zero% = {num_zeros/num_total*100:.2f}%, one% = {num_ones/num_total*100:.2f}%")
        
        return x * gate
    
class DynamicSharingUnit(nn.Module):
    def __init__(self, dim, reduction_ratio=4):
        super().__init__()
        self.dim = dim
        self.reduced_dim = dim // reduction_ratio
        
        # ✅ 输入是 [c_{l-1}, y_l, text]
        self.W1 = nn.Linear(3 * dim, self.reduced_dim)
        
        # Gates
        self.Wc = nn.Linear(self.reduced_dim, dim)
        self.Wi = nn.Linear(self.reduced_dim, dim)
        self.Wf = nn.Linear(self.reduced_dim, dim)
        
        self.bc = nn.Parameter(torch.zeros(dim))
        self.bi = nn.Parameter(torch.zeros(dim))
        self.bf = nn.Parameter(torch.zeros(dim))
        
    def forward(self, y_l, text_global, c_prev):
        # Normalize c_prev
        c_prev_norm = torch.sigmoid(c_prev)
        
        # ✅ Early fusion: 直接concat
        combined = torch.cat([c_prev_norm, y_l, text_global], dim=-1)
        s = F.relu(self.W1(combined))  # [B, D//r]
        
        # Gates
        c_tilde = torch.tanh(self.Wc(s) + self.bc)
        i = torch.sigmoid(self.Wi(s) + self.bi)
        f = torch.sigmoid(self.Wf(s) + self.bf)
        
        # Update
        c_l = f * c_prev + i * c_tilde
        print("c_l max:", c_l.max().item(), "min:", c_l.min().item(), "mean:", c_l.mean().item())
        print("c_tilde max:", c_tilde.max().item(), "min:", c_tilde.min().item(), "mean:", c_tilde.mean().item())
        print("i max:", i.max().item(), "min:", i.min().item(), "mean:", i.mean().item())
        print("f max:", f.max().item(), "min:", f.min().item(), "mean:", f.mean().item())
        
        return c_l


class TextConditionedDynamicLayerAttention(nn.Module):
    """
    严格按照图片公式实现的 Dynamic Layer Attention
    """
    def __init__(self, feature_dim=4096, num_vision_layers=24, 
                 num_heads=8, reduction_ratio=4):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_vision_layers = num_vision_layers
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # ============ 1. DSU for context extraction ============
        self.dsu = DynamicSharingUnit(feature_dim, reduction_ratio)
        
        # ============ 2. g(c_l): 从context生成modulation ============
        self.g = SpatialGate(feature_dim)
        
        # ============ 3. Layer attention ============
        self.W_q = nn.Linear(feature_dim, feature_dim)
        self.W_k = nn.Linear(feature_dim, feature_dim)
        self.W_o = nn.Linear(feature_dim, feature_dim)
        
        # ============ 4. Normalizations ============
        self.q_norm = nn.LayerNorm(feature_dim)
        self.k_norm = nn.LayerNorm(feature_dim)
        self.output_norm = nn.LayerNorm(feature_dim)

        # 可视化
        self.save_counter = 0
        self.save_dir = "/dataset/ca_attention_weights/attention"
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"✓ Cross-Attention save dir: {self.save_dir}")
        self.layer_importance_scores = []
        
        self._reset_parameters()
    

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # ====== 特殊处理：SpatialGate ======
        # 让 gate 初始接近 identity（即 x * 1）
        for m in self.g.conv.modules():
            if isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 0.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
    
    def forward(self, text_features, projected_layer_features):
        """
        Args:
            text_features: [B, T, feature_dim] - 已经过mm_projector
            projected_layer_features: List of [B, N, feature_dim], length=24
        
        Returns:
            attended_output: [B, T, feature_dim]
        """
        B, T, _ = text_features.shape
        N = projected_layer_features[0].shape[1]
        
        # ============ Step 0: 对齐 batch size ============
        B_vision = projected_layer_features[0].shape[0]
        if text_features.shape[0] == 1 and B_vision > 1:
            text_features = text_features.expand(B_vision, -1, -1)
            B = B_vision
        
        # ✅ Step 1: Text全局表示（会在每一层复用）
        text_global = text_features.mean(dim=1)  # [B, feature_dim]
        
        # ✅ Step 2: Forward Path - 逐层提取context
        # c_0 初始化为0（如图片所示）
        c_prev = torch.zeros(B, self.feature_dim, 
                            device=text_features.device, 
                            dtype=text_features.dtype)
        
        contexts = []  # 保存每一层的context c_l
        
        for layer_idx, proj_feat in enumerate(projected_layer_features):
            # y_l = Pool(x^l)
            y_l = proj_feat.mean(dim=1)  # [B, feature_dim]
            
            # c_l = DSU([y_l, text], c_{l-1})
            c_l = self.dsu(y_l, text_global, c_prev)
            
            contexts.append(c_l)
            c_prev = c_l  # 更新为下一层的输入
        
        # ✅ Step 3: Backward Path - 用各自的context刷新每一层
        c_23 = contexts[-2]  # [B, feature_dim]
        pre_23 = projected_layer_features[-2]
        refreshed_23 = self.g(pre_23, c_23)
        print("previous layer 23 max:", pre_23.max().item(), "min:", pre_23.min().item(), "mean:", pre_23.mean().item())
        print("refreshed_23 max:", refreshed_23.max().item(), "min:", refreshed_23.min().item(), "mean:", refreshed_23.mean().item())
        
        # ============ Step 4: Multi-Head Layer Attention ============
        # Q from text
        Q = self.W_q(text_features)  # [B, T, feature_dim]
        Q = self.q_norm(Q)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        print("Q max:", Q.max().item(), "min:", Q.min().item())
        
        # K, V from all refreshed layers
        K = self.W_k(refreshed_23)
        K = self.k_norm(K)
        print("K max:", K.max().item(), "min:", K.min().item())
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = refreshed_23
        print("V max:", V.max().item(), "min:", V.min().item())
        V = V.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        attended = torch.matmul(attn_weights, V)  # [B, h, T, head_dim]
        
        # Reshape and project
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(B, T, self.feature_dim)
        
        output = self.W_o(attended)
        output = self.output_norm(output)
        
        return output
    
def build_text_conditioned_dla(config):
    feature_dim = config.hidden_size  # 4096
    num_vision_layers = getattr(config, 'num_vision_layers', 24)
    num_heads = getattr(config, 'num_heads', 8)
    reduction_ratio = getattr(config, 'reduction_ratio', 4)
    
    return TextConditionedDynamicLayerAttention(
        feature_dim=feature_dim,
        num_vision_layers=num_vision_layers,
        num_heads=num_heads,
        reduction_ratio=reduction_ratio
    )

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')