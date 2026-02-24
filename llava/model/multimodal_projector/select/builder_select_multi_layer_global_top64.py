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
        c_expanded = c_l.unsqueeze(0)

        gate = self.conv(c_expanded + x)
        
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
        s = F.relu(self.W1(combined))  # [D//r]
        
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
                 num_heads=8, reduction_ratio=4, topk=128, focus_layer_idx=-2,):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_vision_layers = num_vision_layers
        self.num_heads = num_heads
        self.topk = topk
        self.head_dim = feature_dim // num_heads
        self.focus_layer_idx = focus_layer_idx
        
        # ============ 1. DSU for context extraction ============
        self.dsu = DynamicSharingUnit(feature_dim, reduction_ratio)

        self.score_q = nn.Linear(feature_dim, feature_dim, bias=False)
        self.score_k = nn.Linear(feature_dim, feature_dim, bias=False)
        self.score_norm = nn.LayerNorm(feature_dim)

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
    
    def forward(self, text_features, projected_layer_features, force_off: bool = False):
        # text_features: [T, D]
        # projected_layer_features: List[L] of [N, D]
        T, D = text_features.shape
        L = len(projected_layer_features)

        # Step 1: text global
        text_global = F.layer_norm(text_features.mean(dim=0), (D,))

        # Step 2: DSU contexts
        c_prev = torch.zeros(self.feature_dim, device=text_features.device, dtype=text_features.dtype)
        contexts = []
        for proj_feat in projected_layer_features:
            y_l = proj_feat.mean(dim=0)  # [D]
            c_prev = self.dsu(y_l, text_global, c_prev)
            contexts.append(c_prev)

        c_final = contexts[-2]  # [D]
        q = self.score_norm(self.score_q(c_final))  # [D]

        # =========================
        # 无保底 + 层内 z-score + 全局 top64
        # =========================
        eps = 1e-6
        final_k = 64

        all_tokens = []
        all_scores = []
        all_layer_ids = []  # optional: 用于 debug / 可视化
        all_pos_ids = []    # optional: token 在该层的位置，用于 debug

        for li, patches in enumerate(projected_layer_features):
            # patches: [N, D]
            N = patches.shape[0]
            if N == 0:
                continue

            k = self.score_norm(self.score_k(patches))          # [N, D]
            scores_raw = (k * q.unsqueeze(0)).sum(dim=-1)       # [N]

            # 层内 z-score 标准化，让层间分数更可比
            mean = scores_raw.mean()
            std = scores_raw.std(unbiased=False)
            scores_z = (scores_raw - mean) / (std + eps)        # [N]

            all_tokens.append(patches)
            all_scores.append(scores_z)

            # debug metadata（可选）
            all_layer_ids.append(torch.full((N,), li, device=patches.device, dtype=torch.long))
            all_pos_ids.append(torch.arange(N, device=patches.device, dtype=torch.long))

        if len(all_tokens) == 0:
            out = torch.empty((0, D), device=text_features.device, dtype=text_features.dtype)
            if force_off:
                out = out * 0.0
            return out

        all_tokens = torch.cat(all_tokens, dim=0)       # [sumN, D]
        all_scores = torch.cat(all_scores, dim=0)       # [sumN]
        all_layer_ids = torch.cat(all_layer_ids, dim=0) # [sumN]
        all_pos_ids = torch.cat(all_pos_ids, dim=0)     # [sumN]


        # 全局 topK（基于 z-score）
        K = min(final_k, all_scores.numel())
        top_scores, top_idx = torch.topk(all_scores, k=K, largest=True, sorted=True)
        evidence_tokens = all_tokens.index_select(0, top_idx)  # [K, D]

        if force_off:
            evidence_tokens = evidence_tokens * 0.0

        # # 你如果想看最终 top64 来自哪些层，可以临时打开：
        # chosen_layers = all_layer_ids.index_select(0, top_idx)
        # print("top64 layer histogram:", torch.bincount(chosen_layers, minlength=L).tolist())
        # print("top64 scores (z):", top_scores[:10].tolist())

        return evidence_tokens

    
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