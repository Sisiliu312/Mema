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

class AnchorResidualLayerMix(nn.Module):
    """
    Token-wise multi-layer mix anchored at layer anchor_idx:
      x_mix = x_anchor + scale * sum_{l!=anchor} alpha_l * (x_l - x_anchor)

    输入: layer_feats: list[L] of [N, D]
    输出: x_mix: [N, D]
    """
    def __init__(self, num_layers: int = 24, anchor_idx: int = -2, init_scale: float = 0.0):
        super().__init__()
        self.num_layers = num_layers
        self.anchor_idx = anchor_idx

        # [L] learnable logits (shared across all tokens)
        self.layer_logits = nn.Parameter(torch.zeros(num_layers))

        # scalar scale (shape = [1])
        self.scale = nn.Parameter(torch.tensor([float(init_scale)]))

    def forward(self, layer_feats):
        assert isinstance(layer_feats, (list, tuple)) and len(layer_feats) == self.num_layers
        assert layer_feats[0].dim() == 2, f"expect [N,D], got {layer_feats[0].shape}"

        # anchor: [N, D]
        x_anchor = layer_feats[self.anchor_idx]

        # stack -> [L, N, D]
        x = torch.stack(layer_feats, dim=0)

        # diff -> [L, N, D]
        diff = x - x_anchor.unsqueeze(0)

        # anchor position (0..L-1)
        anchor_pos = self.anchor_idx if self.anchor_idx >= 0 else (self.num_layers + self.anchor_idx)

        # mask anchor layer
        mask = torch.ones(self.num_layers, device=self.layer_logits.device, dtype=torch.bool)
        mask[anchor_pos] = False

        # masked softmax in fp32 (stable)
        logits_fp32 = self.layer_logits.float()                    # [L]
        masked_logits = logits_fp32.masked_fill(~mask, float("-inf"))
        alpha_fp32 = F.softmax(masked_logits, dim=0)               # [L]

        # cast + reshape for broadcast -> [L,1,1]
        alpha = alpha_fp32.to(diff.dtype).view(self.num_layers, 1, 1)

        # weighted sum over layers -> [N, D]
        delta = torch.sum(alpha * diff, dim=0)

        # scale dtype align
        scale = self.scale.to(delta.dtype)                         # scalar
        x_mix = x_anchor + scale * delta
        return x_mix
    
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
                 num_heads=8, reduction_ratio=4, topk=64):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_vision_layers = num_vision_layers
        self.num_heads = num_heads
        self.topk = topk
        self.head_dim = feature_dim // num_heads
        
        # ============ 1. DSU for context extraction ============
        self.dsu = DynamicSharingUnit(feature_dim, reduction_ratio)

        self.score_q = nn.Linear(feature_dim, feature_dim, bias=False)
        self.score_k = nn.Linear(feature_dim, feature_dim, bias=False)
        self.score_norm = nn.LayerNorm(feature_dim)

        self.layer_mix = AnchorResidualLayerMix(num_layers=24, anchor_idx=-2, init_scale=0.0)

        # # 可视化
        # self.save_counter = 0
        # self.save_dir = "/dataset/ca_attention_weights/attention"
        # os.makedirs(self.save_dir, exist_ok=True)
        # print(f"✓ Cross-Attention save dir: {self.save_dir}")
        # self.layer_importance_scores = []
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for m in self.dsu.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _group_pool(self, fea, grid_size=24, groups_per_side=6):
        # fea: [N, D], N=grid_size^2
        N, D = fea.shape
        assert N == grid_size * grid_size, f"N={N} not match {grid_size}x{grid_size}"
        x = fea.view(grid_size, grid_size, D)
        gs = grid_size // groups_per_side
        x = x.view(groups_per_side, gs, groups_per_side, gs, D)  # [Gp,gs,Gp,gs,D]
        x = x.permute(0, 2, 1, 3, 4).contiguous()               # [Gp,Gp,gs,gs,D]
        groups = x.view(groups_per_side * groups_per_side, gs * gs, D).mean(dim=1)  # [G,D]
        return groups
    
    def forward(self, text_features, projected_layer_features, force_off: bool = False):
        """
        Args:
            text_features: [B, T, feature_dim] - 已经过mm_projector
            projected_layer_features: List of [B, N, feature_dim], length=24
        
        Returns:
            attended_output: [B, T, feature_dim]
        """
        T, D = text_features.shape
        N, _ = projected_layer_features[0].shape
        
        # ✅ Step 1: Text全局表示（会在每一层复用）
        text_global = F.layer_norm(text_features.mean(dim=0), (D,))
        
        # ✅ Step 2: Forward Path - 逐层提取context
        # c_0 初始化为0（如图片所示）
        c_prev = torch.zeros(self.feature_dim, 
                            device=text_features.device, 
                            dtype=text_features.dtype)
        
        contexts = []  # 保存每一层的context c_l
        
        for layer_idx, proj_feat in enumerate(projected_layer_features):
            # y_l = Pool(x^l)
            y_l = proj_feat.mean(dim=0)  # [B, feature_dim]
            
            # c_l = DSU([y_l, text], c_{l-1})
            c_l = self.dsu(y_l, text_global, c_prev)
            
            contexts.append(c_l)
            c_prev = c_l  # 更新为下一层的输入
        
        c_final = contexts[-2]                      # [D]
        img_tokens = self.layer_mix(projected_layer_features)  # [M, N, D]


        q = self.score_norm(self.score_q(c_final))                     # [D]
        k = self.score_norm(self.score_k(img_tokens))           # [N,D]
        scores = (k * q.unsqueeze(0)).sum(dim=-1)                      # [N]

        # 6) top-k 证据选择（离散瓶颈）
        K = min(self.topk, N)
        top_scores, top_idx = torch.topk(scores, k=K, dim=0, largest=True, sorted=True)

        evidence_tokens = img_tokens.index_select(0, top_idx)   # [K,D]
        # print("evidence_tokens shape:", evidence_tokens.shape)

        if force_off:
            evidence_tokens = evidence_tokens * 0.0
        
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