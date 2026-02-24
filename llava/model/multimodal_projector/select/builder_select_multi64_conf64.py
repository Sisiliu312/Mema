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
        device = text_features.device
        dtype = text_features.dtype
        eps = 1e-6

        # DSU contexts
        text_global = F.layer_norm(text_features.mean(dim=0), (D,))
        c_prev = torch.zeros(self.feature_dim, device=device, dtype=dtype)
        contexts = []
        for proj_feat in projected_layer_features:
            y_l = proj_feat.mean(dim=0)
            c_prev = self.dsu(y_l, text_global, c_prev)
            contexts.append(c_prev)

        # 单一 query
        c_final = contexts[-2]
        q = self.score_norm(self.score_q(c_final))  # [D]

        # ---- 超参 ----
        cand_total = 128
        final_k = 64
        focus = self.focus_layer_idx % L            # focus=22(第23层)
        focus_cand = 64                             # ✅候选池中 focus 固定取64
        other_cand = cand_total - focus_cand        # 64

        focus_keep = 32                             # ✅最终输出中 focus 固定取32
        other_keep = final_k - focus_keep           # 32

        # ---- 先算每层 raw scores & conf(z_max) ----
        scores_raw_list = []
        conf_list = []
        for li, patches in enumerate(projected_layer_features):
            if patches.numel() == 0:
                scores_raw_list.append(torch.empty((0,), device=device, dtype=dtype))
                conf_list.append(torch.tensor(-1e9, device=device, dtype=torch.float32))
                continue

            k = self.score_norm(self.score_k(patches))            # [N,D]
            scores_raw = (k * q.unsqueeze(0)).sum(dim=-1)         # [N]
            scores_raw_list.append(scores_raw)

            mean = scores_raw.mean()
            std = scores_raw.std(unbiased=False) + eps
            z_max = ((scores_raw.max() - mean) / std).float()
            conf_list.append(z_max)

        conf = torch.stack(conf_list)  # [L]

        # =========================================================
        # (1) 构造候选池：focus层64 + 其他层按conf分配64
        # =========================================================
        k_plan = [0] * L
        # focus 固定取64（cap 到该层 N）
        k_plan[focus] = min(focus_cand, projected_layer_features[focus].shape[0])

        # 其他层 softmax(conf) 分配 other_cand=64
        other_layers = [i for i in range(L) if i != focus]
        if other_cand > 0 and len(other_layers) > 0:
            w = torch.softmax(conf[other_layers], dim=0)  # [L-1]
            alloc = torch.floor(w * other_cand).long()
            rem = int(other_cand - alloc.sum().item())
            if rem > 0:
                order = torch.argsort(w, descending=True)
                for j in order.tolist():
                    if rem <= 0:
                        break
                    alloc[j] += 1
                    rem -= 1

            for jj, li in enumerate(other_layers):
                k_plan[li] = min(int(alloc[jj].item()), projected_layer_features[li].shape[0])

        # ---- per-layer topk 形成候选池 ----
        cand_tokens = []
        cand_scores = []
        cand_layer_ids = []
        for li, patches in enumerate(projected_layer_features):
            k_li = int(k_plan[li])
            if k_li <= 0:
                continue
            scores_raw = scores_raw_list[li]
            if scores_raw.numel() == 0:
                continue

            k_li = min(k_li, scores_raw.numel())
            top_scores, top_idx = torch.topk(scores_raw, k=k_li, largest=True, sorted=True)
            cand_tokens.append(patches.index_select(0, top_idx))  # [k_li, D]
            cand_scores.append(top_scores)                        # [k_li]
            cand_layer_ids.append(torch.full((k_li,), li, device=device, dtype=torch.long))

        if len(cand_tokens) == 0:
            out = torch.empty((0, D), device=device, dtype=dtype)
            if force_off:
                out = out * 0.0
            return out

        cand_tokens = torch.cat(cand_tokens, dim=0)      # [Kcand, D] 期望约128
        cand_scores = torch.cat(cand_scores, dim=0)      # [Kcand]
        cand_layer_ids = torch.cat(cand_layer_ids, dim=0)# [Kcand]

        # =========================================================
        # (2) 二次筛选：focus里取32 + (剩余候选池)里取top32
        # =========================================================
        # 找出候选池里属于focus层的索引
        focus_mask = (cand_layer_ids == focus)
        focus_idx = torch.nonzero(focus_mask, as_tuple=False).squeeze(-1)

        selected_idx_parts = []

        # 2.1 focus 保底取 focus_keep=32（按 cand_scores）
        if focus_idx.numel() > 0 and focus_keep > 0:
            focus_scores = cand_scores.index_select(0, focus_idx)
            k1 = min(int(focus_keep), int(focus_scores.numel()))
            _, ord1 = torch.topk(focus_scores, k=k1, largest=True, sorted=True)
            sel_focus = focus_idx.index_select(0, ord1)
            selected_idx_parts.append(sel_focus)
        else:
            k1 = 0

        # 2.2 剩余名额 other_keep=32：从剩余候选池里取 top32
        remain = other_keep
        if remain > 0:
            if len(selected_idx_parts) > 0:
                already = selected_idx_parts[0]
                keep_mask = torch.ones(cand_scores.shape[0], device=device, dtype=torch.bool)
                keep_mask[already] = False
                rest_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
            else:
                rest_idx = torch.arange(cand_scores.shape[0], device=device)

            rest_scores = cand_scores.index_select(0, rest_idx)
            k2 = min(int(remain), int(rest_scores.numel()))
            _, ord2 = torch.topk(rest_scores, k=k2, largest=True, sorted=True)
            sel_rest = rest_idx.index_select(0, ord2)
            selected_idx_parts.append(sel_rest)

        selected_idx = torch.cat(selected_idx_parts, dim=0) if len(selected_idx_parts) > 0 else torch.empty((0,), device=device, dtype=torch.long)

        # 可选：最终按分数再排序（更稳定）
        sel_scores = cand_scores.index_select(0, selected_idx)
        sort_final = torch.argsort(sel_scores, descending=True)
        selected_idx = selected_idx.index_select(0, sort_final)

        evidence_tokens = cand_tokens.index_select(0, selected_idx)  # [<=64, D]

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