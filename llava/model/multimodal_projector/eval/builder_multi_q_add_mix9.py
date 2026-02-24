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


class TextConditionedDynamicLayerAttention(nn.Module):
    """
    严格按照图片公式实现的 Dynamic Layer Attention
    """
    def __init__(self, feature_dim=4096, num_vision_layers=24, 
                 num_heads=8, reduction_ratio=4, topk=64, focus_layer_idx=-2,):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_vision_layers = num_vision_layers
        self.num_heads = num_heads
        self.topk = topk
        self.head_dim = feature_dim // num_heads
        self.focus_layer_idx = focus_layer_idx
        
        # ============ 1. DSU for context extraction ============
        self.dsu = DynamicSharingUnit(feature_dim, reduction_ratio)
        self.layer_mix = AnchorResidualLayerMix(num_layers=24, anchor_idx=-2, init_scale=0.0)

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

    @staticmethod
    def _alloc_topk(total_k: int, L: int, focus_idx: int):
        """
        分配规则：
          focus 层拿一半（floor(total_k/2)）
          其余层均分另一半（尽量均匀，余数从前往后补）
        返回：k_per_layer: List[int] 长度 L
        """
        total_k = int(max(0, total_k))
        if total_k == 0:
            return [0] * L

        focus_k = total_k // 2
        rest_k = total_k - focus_k  # 另一半（含奇数时多出来的1）

        other_layers = L - 1
        base = rest_k // other_layers
        rem = rest_k - base * other_layers

        k_per_layer = [base] * L
        k_per_layer[focus_idx] = focus_k

        # 把余数 rem 分给非 focus 层（从小层号开始补）
        for li in range(L):
            if li == focus_idx:
                continue
            if rem <= 0:
                break
            k_per_layer[li] += 1
            rem -= 1

        return k_per_layer
    
    def forward(self, text_features, projected_layer_features, force_off: bool = False):
        T, D = text_features.shape
        L = len(projected_layer_features)

        text_global = F.layer_norm(text_features.mean(dim=0), (D,))

        # contexts: c_l
        c_prev = torch.zeros(self.feature_dim, device=text_features.device, dtype=text_features.dtype)
        contexts = []
        for proj_feat in projected_layer_features:
            y_l = proj_feat.mean(dim=0)
            c_prev = self.dsu(y_l, text_global, c_prev)
            contexts.append(c_prev)

        # 每层一个 query：q_l = score_q(c_l)
        q_list = [self.score_norm(self.score_q(c_l)) for c_l in contexts]  # List[L] of [D]

        # 原先固定配额
        focus = self.focus_layer_idx % L
        k_plan = self._alloc_topk(self.topk, L, focus)

        evidence_list = []
        score_list = []

        for li, patches in enumerate(projected_layer_features):
            N = patches.shape[0]
            k_li = min(k_plan[li], N)
            if k_li <= 0:
                continue

            k = self.score_norm(self.score_k(patches))          # [N, D]
            scores = (k * q_list[li].unsqueeze(0)).sum(dim=-1)  # ✅用该层 q_l

            top_scores, top_idx = torch.topk(scores, k=k_li, largest=True, sorted=True)
            evidence = patches.index_select(0, top_idx)

            evidence_list.append(evidence)
            score_list.append(top_scores)

        if len(evidence_list) == 0:
            evidence_tokens = torch.empty((0, D), device=text_features.device, dtype=text_features.dtype)
        else:
            evidence_tokens = torch.cat(evidence_list, dim=0)

        print("evidence_tokens shape:", evidence_tokens.shape)

        img_tokens = self.layer_mix(projected_layer_features)  # [M, N, D]
        groups = self._group_pool(img_tokens, 24, 3)  # [G,D]
        attn_tokens_groups = torch.cat([evidence_tokens, groups], dim=0)  # [M+G, D]
        print("attn_tokens_groups shape:", attn_tokens_groups.shape)

        if force_off:
            attn_tokens_groups = attn_tokens_groups * 0.0

        return attn_tokens_groups

    
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