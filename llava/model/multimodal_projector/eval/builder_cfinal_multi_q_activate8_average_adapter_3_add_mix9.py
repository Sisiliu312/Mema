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

        self.score_q = nn.Linear(feature_dim, feature_dim, bias=False)
        self.score_k = nn.Linear(feature_dim, feature_dim, bias=False)
        self.score_norm = nn.LayerNorm(feature_dim)
        self.layer_mix = AnchorResidualLayerMix(num_layers=24, anchor_idx=-2, init_scale=0.0)

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

    def forward(self, text_features, projected_layer_features, force_off: bool = False):
        # text_features: [T, D]
        # projected_layer_features: List[L] of [N, D]
        T, D = text_features.shape
        L = len(projected_layer_features)
        device = text_features.device
        dtype = text_features.dtype
        eps = 1e-6

        focus = self.focus_layer_idx % L

        # ===== 1) DSU contexts =====
        text_global = F.layer_norm(text_features.mean(dim=0), (D,))
        c_prev = torch.zeros(self.feature_dim, device=device, dtype=dtype)
        contexts = []
        for proj_feat in projected_layer_features:
            y_l = proj_feat.mean(dim=0)
            c_prev = self.dsu(y_l, text_global, c_prev)
            contexts.append(c_prev)

        # 每层 query：q_l
        q_list = [self.score_norm(self.score_q(c_l)) for c_l in contexts]  # List[L] of [D]

        # ===== 2) 每层 scores + conf（都用 q_l）=====
        layer_scores = []
        layer_conf = []
        for li, patches in enumerate(projected_layer_features):
            # patches: [N, D]
            if patches.numel() == 0:
                layer_scores.append(torch.empty((0,), device=device, dtype=dtype))
                layer_conf.append(torch.tensor(-1e9, device=device, dtype=torch.float32))
                continue

            k = self.score_norm(self.score_k(patches))  # [N, D]
            scores_raw = (k * q_list[li].unsqueeze(0)).sum(dim=-1)  # [N]

            # 层内 z-score（推荐开；否则 conf 阈值意义不稳定）
            mean = scores_raw.mean()
            std = scores_raw.std(unbiased=False) + eps
            scores = (scores_raw - mean) / std

            layer_scores.append(scores)

            # conf：峰值 z（越大越说明“这层确实有证据”）
            layer_conf.append(scores.max().float())

        layer_conf = torch.stack(layer_conf)  # [L]

        # ===== 3) 分配配额：严格 focus32 + nonfocus32 =====
        final_focus = 32
        final_nonfocus = 32

        # focus 候选数（先从 focus 层拿 32）
        focus_scores = layer_scores[focus]
        focus_N = focus_scores.numel()
        focus_take = min(final_focus, focus_N)

        # nonfocus：先门控出“可用层集合”，然后均分 32
        thresh = 3.0  # 你可以调成 2.5 / 2.0 更保守
        nonfocus_mask = torch.ones(L, device=device, dtype=torch.bool)
        nonfocus_mask[focus] = False
        active_mask = nonfocus_mask & (layer_conf > thresh)

        active_layers = torch.nonzero(active_mask, as_tuple=False).squeeze(-1)  # tensor idx

        # fallback：如果通过门控的层太少，会导致覆盖崩；给一个保底策略
        # 这里我建议：至少保证 active >= 8（你也可以用 12）
        min_active = 8
        if active_layers.numel() < min_active:
            # 取 nonfocus 中 conf 最高的若干层来补足 active 集合
            cand_layers = torch.nonzero(nonfocus_mask, as_tuple=False).squeeze(-1)
            cand_conf = layer_conf.index_select(0, cand_layers)
            _, order = torch.topk(cand_conf, k=min(min_active, cand_layers.numel()), largest=True, sorted=True)
            active_layers = cand_layers.index_select(0, order)

        M = int(active_layers.numel())
        # 均分 nonfocus 32
        base = final_nonfocus // M
        rem = final_nonfocus - base * M

        k_per_layer = [0] * L
        k_per_layer[focus] = focus_take
        # 给 active 层分配
        for idx_i, li in enumerate(active_layers.tolist()):
            k_li = base + (1 if idx_i < rem else 0)
            k_per_layer[li] = min(k_li, layer_scores[li].numel())

        # 如果某些层 N 太小导致没拿够 32，再从 active_layers 里按分数补齐
        got_nonfocus = sum(k_per_layer[li] for li in range(L) if li != focus)
        deficit = final_nonfocus - got_nonfocus
        if deficit > 0:
            # 逐个从 active 层里加 1，优先给“还能加”的层
            # 这里用每层剩余容量 = N - 当前k
            remain_cap = []
            for li in active_layers.tolist():
                cap = int(layer_scores[li].numel()) - int(k_per_layer[li])
                remain_cap.append(cap)
            remain_cap = torch.tensor(remain_cap, device=device)
            # 依次补齐
            order = torch.argsort(remain_cap, descending=True)
            for j in order.tolist():
                if deficit <= 0:
                    break
                li = int(active_layers[j].item())
                if k_per_layer[li] < layer_scores[li].numel():
                    k_per_layer[li] += 1
                    deficit -= 1

        # ===== 4) 逐层取 topk，并严格拼出 focus32 + nonfocus32 =====
        # focus tokens
        evidence_focus = []
        if focus_take > 0:
            s = layer_scores[focus]
            patches = projected_layer_features[focus]
            _, idx = torch.topk(s, k=focus_take, largest=True, sorted=True)
            evidence_focus.append(patches.index_select(0, idx))
        evidence_focus = torch.cat(evidence_focus, dim=0) if len(evidence_focus) else torch.empty((0, D), device=device, dtype=dtype)

        # nonfocus tokens
        evidence_nonfocus = []
        for li in range(L):
            if li == focus:
                continue
            k_li = int(k_per_layer[li])
            if k_li <= 0:
                continue
            s = layer_scores[li]
            patches = projected_layer_features[li]
            _, idx = torch.topk(s, k=k_li, largest=True, sorted=True)
            evidence_nonfocus.append(patches.index_select(0, idx))
        evidence_nonfocus = torch.cat(evidence_nonfocus, dim=0) if len(evidence_nonfocus) else torch.empty((0, D), device=device, dtype=dtype)

        # 严格截断到 32（防止因为补齐逻辑导致>32）
        if evidence_nonfocus.shape[0] > final_nonfocus:
            # 直接按各自层内的 topk 已经是强证据了；这里就保持原顺序截断即可
            evidence_nonfocus = evidence_nonfocus[:final_nonfocus]

        # 严格拼接 focus32 + nonfocus32
        evidence_tokens = torch.cat([evidence_focus, evidence_nonfocus], dim=0)
        # print("evidence_tokens shape:", evidence_tokens.shape)


        img_tokens = self.layer_mix(projected_layer_features)  # [M, N, D]
        groups = self._group_pool(img_tokens, 24, 3)  # [G,D]
        attn_tokens_groups = torch.cat([evidence_tokens, groups], dim=0)  # [M+G, D]
        # print("attn_tokens_groups shape:", attn_tokens_groups.shape)

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