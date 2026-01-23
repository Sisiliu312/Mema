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
        c_prev_used = c_prev.detach()
        c_prev_norm = torch.sigmoid(c_prev_used)
        
        # ✅ Early fusion: 直接concat
        combined = torch.cat([c_prev_norm, y_l, text_global], dim=-1)
        s = F.relu(self.W1(combined))  # [B, D//r]
        
        # Gates
        c_tilde = torch.tanh(self.Wc(s) + self.bc)
        i = torch.sigmoid(self.Wi(s) + self.bi)
        f = torch.sigmoid(self.Wf(s) + self.bf)
        
        # Update
        c_l = f * c_prev + i * c_tilde
        
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
        for m in self.dsu.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
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
        refreshed_features = []
        
        for layer_idx, proj_feat in enumerate(projected_layer_features):
            # ✅ 关键：用当前层的 c_l，不是 c_final！
            c_l = contexts[layer_idx]  # [B, feature_dim]
            
            # refreshed_feat = self.g(proj_feat, c_l)
            refreshed_feat = self.g(proj_feat, c_l.detach())
            refreshed_features.append(refreshed_feat)
        
        # ============ Step 4: Multi-Head Layer Attention ============
        # Q from text
        Q = self.W_q(text_features)  # [B, T, feature_dim]
        Q = self.q_norm(Q)
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        # K, V from all refreshed layers
        K_list = []
        V_list = []
        
        for refreshed_feat in refreshed_features:
            K = self.W_k(refreshed_feat)
            K = self.k_norm(K)
            K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            V = refreshed_feat.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
            
            K_list.append(K)
            V_list.append(V)
        
        # Concatenate all layers
        K_all = torch.cat(K_list, dim=2)  # [B, num_heads, 24*N, head_dim]
        V_all = torch.cat(V_list, dim=2)
        
        # Attention
        attn_scores = torch.matmul(Q, K_all.transpose(-2, -1))
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        extra_scale = (24 ** 0.5)  # sqrt(24) ≈ 4.9
        attn_scores = attn_scores / extra_scale
        attn_weights = F.softmax(attn_scores, dim=-1)


        # ============ 步骤6: 可视化 ============
        if not self.training:
            self._visualize_per_image(attn_weights, N)
        
        attended = torch.matmul(attn_weights, V_all)  # [B, h, T, head_dim]
        
        # Reshape and project
        attended = attended.transpose(1, 2).contiguous()
        attended = attended.view(B, T, self.feature_dim)
        
        output = self.W_o(attended)
        output = self.output_norm(output)
        
        return output
    
    def _visualize_per_image(self, attn_weights, N):
        """
        逐图保存 layer importance
        
        Args:
            attn_weights: [B, num_heads, T, 24*N]
            N: 每层的 token 数量
        """
        B = attn_weights.shape[0]
        
        # 平均所有 heads
        attn_weights_avg = attn_weights.mean(dim=1)  # [B, T, 24*N]
        
        for batch_idx in range(B):
            layer_importances = []
            
            for layer_idx in range(24):
                start_idx = layer_idx * N
                end_idx = (layer_idx + 1) * N
                
                # 该图该层的所有 attention
                attn_to_layer = attn_weights_avg[batch_idx, :, start_idx:end_idx]  # [T, N]
                
                # ✅ 方案1: 平均所有 text tokens 和 vision tokens
                importance = attn_to_layer.mean().item()
                
                # ✅ 方案2: 对每个 text token，取其最关注的 vision token，再平均
                # importance = attn_to_layer.max(dim=-1).values.mean().item()
                
                layer_importances.append(importance)
            
            # 保存单张图的 layer importance
            save_path = os.path.join(
                self.save_dir, 
                f"layer_importance_{self.save_counter:06d}.npy"
            )
            np.save(save_path, np.array(layer_importances))
            
            self.save_counter += 1
    
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