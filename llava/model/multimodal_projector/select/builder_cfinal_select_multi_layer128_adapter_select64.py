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

        text_global = F.layer_norm(text_features.mean(dim=0), (D,))

        c_prev = torch.zeros(self.feature_dim, device=text_features.device, dtype=text_features.dtype)
        contexts = []
        for proj_feat in projected_layer_features:
            y_l = proj_feat.mean(dim=0)
            c_prev = self.dsu(y_l, text_global, c_prev)
            contexts.append(c_prev)

        c_final = contexts[-2]
        q = self.score_norm(self.score_q(c_final))  # [D]

        focus = self.focus_layer_idx % L

        # =========================
        # 改法1：门控（质量驱动）
        # =========================
        eps = 1e-6
        thresh = 2.0  # ✅ 门控阈值：可扫 1.5/2.0/2.5

        # 候选池大小（建议比最终输出大）
        # 你可以把 self.topk 设成 128（候选），final_k 再裁到 64/80
        cand_total = int(self.topk)

        # 最终输出大小（安全区）
        final_k = 64  # ✅ 建议 64 或 80
        focus_min = final_k // 2  # ✅ focus 层最终保底

        # 给 focus 层在候选阶段也保底（避免候选就不够）
        focus_cand_min = min(cand_total // 2, projected_layer_features[focus].shape[0])

        # 1) 先算每层 scores_z 和 conf
        layer_scores_z = []
        layer_conf = []
        for li, patches in enumerate(projected_layer_features):
            k = self.score_norm(self.score_k(patches))
            scores = (k * q.unsqueeze(0)).sum(dim=-1)

            # 层内标准化，让 conf 可比
            scores_z = (scores - scores.mean()) / (scores.std(unbiased=False) + eps)
            layer_scores_z.append(scores_z)
            layer_conf.append(scores_z.max())

        layer_conf = torch.stack(layer_conf)  # [L]

        # 2) 选出通过门控的层（focus 永远通过）
        active = []
        for li in range(L):
            if li == focus:
                active.append(li)
            else:
                if layer_conf[li].item() > thresh:
                    active.append(li)

        # 3) 分配候选预算：focus 先拿 focus_cand_min，其余按 conf softmax 分配剩余
        k_per_layer = [0] * L
        k_per_layer[focus] = focus_cand_min
        rest_budget = cand_total - focus_cand_min
        if rest_budget < 0:
            rest_budget = 0

        other = [li for li in active if li != focus]
        if len(other) > 0 and rest_budget > 0:
            conf_vals = layer_conf[other]
            weights = torch.softmax(conf_vals, dim=0)
            alloc = torch.floor(weights * rest_budget).long()

            # 处理舍入余数
            rem = int(rest_budget - alloc.sum().item())
            if rem > 0:
                order = torch.argsort(weights, descending=True)
                for j in order.tolist():
                    if rem <= 0:
                        break
                    alloc[j] += 1
                    rem -= 1

            for j, li in enumerate(other):
                k_li = int(alloc[j].item())
                if k_li <= 0:
                    continue
                k_per_layer[li] = min(k_li, layer_scores_z[li].numel())

        # =========================
        # 候选采样（按门控+预算）
        # =========================
        evidence_list = []
        score_list = []
        layer_id_list = []

        for li in range(L):
            k_li = k_per_layer[li]
            if k_li <= 0:
                continue

            scores_z = layer_scores_z[li]
            patches = projected_layer_features[li]
            k_li = min(k_li, patches.shape[0])
            if k_li <= 0:
                continue

            top_scores, top_idx = torch.topk(scores_z, k=k_li, largest=True, sorted=True)
            evidence = patches.index_select(0, top_idx)

            evidence_list.append(evidence)
            score_list.append(top_scores)
            layer_id_list.append(torch.full((k_li,), li, device=patches.device, dtype=torch.long))

        if len(evidence_list) == 0:
            evidence_tokens = torch.empty((0, D), device=text_features.device, dtype=text_features.dtype)
            if force_off:
                evidence_tokens = evidence_tokens * 0.0
            print("evidence_tokens shape:", evidence_tokens.shape)
            return evidence_tokens

        evidence_tokens = torch.cat(evidence_list, dim=0)
        all_scores = torch.cat(score_list, dim=0)
        all_layer_ids = torch.cat(layer_id_list, dim=0)
        print("evidence_tokens shape:", evidence_tokens.shape)

        # =========================
        # 改法2：二次筛选（最终输出 topK + focus 保底）
        # =========================
        final_k = min(final_k, evidence_tokens.shape[0])
        focus_min = min(focus_min, final_k)

        # focus 保底
        focus_mask = (all_layer_ids == focus)
        focus_idx = torch.nonzero(focus_mask, as_tuple=False).squeeze(-1)

        selected_idx = []
        if focus_idx.numel() > 0 and focus_min > 0:
            focus_scores = all_scores.index_select(0, focus_idx)
            _, order = torch.topk(focus_scores, k=min(focus_min, focus_scores.numel()), largest=True, sorted=True)
            selected_focus = focus_idx.index_select(0, order)
            selected_idx.append(selected_focus)

        remain = final_k - (selected_idx[0].numel() if len(selected_idx) > 0 else 0)
        if remain > 0:
            if len(selected_idx) > 0:
                already = selected_idx[0]
                keep_mask = torch.ones(all_scores.shape[0], device=all_scores.device, dtype=torch.bool)
                keep_mask[already] = False
                cand_idx = torch.nonzero(keep_mask, as_tuple=False).squeeze(-1)
            else:
                cand_idx = torch.arange(all_scores.shape[0], device=all_scores.device)

            cand_scores = all_scores.index_select(0, cand_idx)
            _, order = torch.topk(cand_scores, k=min(remain, cand_scores.numel()), largest=True, sorted=True)
            selected_rest = cand_idx.index_select(0, order)
            selected_idx.append(selected_rest)

        selected_idx = torch.cat(selected_idx, dim=0) if len(selected_idx) > 0 else torch.empty((0,), device=all_scores.device, dtype=torch.long)

        # 最终按分数排序
        sel_scores = all_scores.index_select(0, selected_idx)
        sort2 = torch.argsort(sel_scores, descending=True)
        selected_idx = selected_idx.index_select(0, sort2)

        evidence_tokens = evidence_tokens.index_select(0, selected_idx)

        if force_off:
            evidence_tokens = evidence_tokens * 0.0

        print("evidence_tokens shape:", evidence_tokens.shape)
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