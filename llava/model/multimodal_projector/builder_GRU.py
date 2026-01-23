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
        self.delta = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))  # 初始关闭注入

    def forward(self, x, c_l):
        """
        x:   [N, D]
        c_l: [D]
        """
        c = c_l.unsqueeze(0).expand_as(x)  # [N,D]
        h = F.layer_norm(x + c, (x.size(-1),))
        delta = torch.tanh(self.delta(h))  # [-1,1] 限幅增量，防止V爆

        a = torch.tanh(self.alpha)         # [-1,1]，初始≈0
        return x + a * delta               # ✅ 初始等价于 x


class DynamicSharingUnit(nn.Module):
    def __init__(self, dim, num_layers=24, z_tau=3.0, init_z_bias=-2.0):
        super().__init__()
        self.dim = dim
        self.num_layers = num_layers
        self.z_tau = float(z_tau)

        self.layer_embed = nn.Embedding(num_layers, dim)
        self.Wc = nn.Linear(2 * dim, dim)     # (y_l, e_l) -> c_tilde
        self.Wz = nn.Linear(4 * dim, dim)

        nn.init.zeros_(self.Wz.weight)
        nn.init.constant_(self.Wz.bias, init_z_bias)

    def forward(self, y_l, text_global, c_prev, layer_idx: int):

        idx = torch.tensor(layer_idx, device=y_l.device, dtype=torch.long)
        e_l = self.layer_embed(idx)  # [D]


        c_tilde = torch.tanh(self.Wc(torch.cat([y_l, e_l], dim=-1)))        # [D]
        z_logit = self.Wz(torch.cat([y_l, c_prev, e_l, text_global], dim=-1))  # [1]
        z = torch.sigmoid(z_logit / self.z_tau)                                # scalar

        c_l = c_prev + z * (c_tilde - c_prev)                              # [D]
        return c_l


class TextConditionedDynamicLayerAttention(nn.Module):
    def __init__(self, feature_dim=4096, num_vision_layers=24, num_heads=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_vision_layers = num_vision_layers
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads

        self.dsu = DynamicSharingUnit(dim=feature_dim, num_layers=num_vision_layers, z_tau=3.0, init_z_bias=-2.0)
        self.g = SpatialGate(feature_dim)

        self.W_q = nn.Linear(feature_dim, feature_dim)
        self.W_k = nn.Linear(feature_dim, feature_dim)
        self.W_o = nn.Linear(feature_dim, feature_dim)

        self.q_norm = nn.LayerNorm(feature_dim)
        self.k_norm = nn.LayerNorm(feature_dim)
        self.v_norm = nn.LayerNorm(feature_dim)
        self.output_norm = nn.LayerNorm(feature_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        nn.init.constant_(self.g.conv[0].weight, 0.0)
        nn.init.constant_(self.g.conv[0].bias, 0.0)
        nn.init.constant_(self.g.conv[2].weight, 0.0)
        nn.init.constant_(self.g.conv[2].bias, 2.0)

        nn.init.zeros_(self.dsu.Wz.weight)
        nn.init.constant_(self.dsu.Wz.bias, -2.0)

    def forward(self, text_features, projected_layer_features):
        """
        text_features: [T, D]
        projected_layer_features: list(24) of [N, D]
        return: [T, D]
        """
        T, D = text_features.shape
        N, _ = projected_layer_features[0].shape

        # 1) text_global: [D]
        text_global = F.layer_norm(text_features.mean(dim=0), (self.feature_dim,))

        # 2) DSU contexts: each [D]
        c_prev = torch.zeros(D, device=text_features.device, dtype=text_features.dtype)
        contexts = []
        for layer_idx, proj_feat in enumerate(projected_layer_features):
            # proj_feat: [N, D] -> y_l: [D]
            y_l = proj_feat.mean(dim=0)
            c_prev = self.dsu(y_l, text_global, c_prev, layer_idx)
            contexts.append(c_prev)

        # 3) refresh layer-23 (index -2)
        c_23 = contexts[-2]                      # [D]
        pre_23 = projected_layer_features[-2]    # [N, D]
        refreshed_23 = self.g(pre_23, c_23)      # [N, D]

        # 4) attention: Q from text, K/V from refreshed_23
        Q = self.q_norm(self.W_q(text_features))     # [T, D]
        K = self.k_norm(self.W_k(refreshed_23))      # [N, D]
        V = self.v_norm(refreshed_23)                              # [N, D]

        H, Hd = self.num_heads, self.head_dim

        # [T,D] -> [1,H,T,Hd]
        q = Q.view(T, H, Hd).transpose(0, 1).unsqueeze(0)
        # [N,D] -> [1,H,N,Hd]
        k = K.view(N, H, Hd).transpose(0, 1).unsqueeze(0)
        v = V.view(N, H, Hd).transpose(0, 1).unsqueeze(0)

        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=False)
        # [1,H,T,Hd] -> [T,D]
        attn = attn.squeeze(0).transpose(1, 2).contiguous()  # [H,T,Hd]
        attn = attn.transpose(0, 1).contiguous().view(T, D)

        out = self.output_norm(self.W_o(attn))
        return out

    
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
    
    return TextConditionedDynamicLayerAttention(
        feature_dim=feature_dim,
        num_vision_layers=num_vision_layers,
        num_heads=num_heads
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