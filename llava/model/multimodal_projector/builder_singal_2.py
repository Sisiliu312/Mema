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
        # c_prev_norm = torch.sigmoid(c_prev)
        c_prev_norm = F.layer_norm(c_prev, (c_prev.shape[-1],))
        
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
        self.v_norm = nn.LayerNorm(feature_dim)
        self.output_norm = nn.LayerNorm(feature_dim)

        # # 可视化
        # self.save_counter = 0
        # self.save_dir = "/dataset/ca_attention_weights/attention"
        # os.makedirs(self.save_dir, exist_ok=True)
        # print(f"✓ Cross-Attention save dir: {self.save_dir}")
        # self.layer_importance_scores = []
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        """
        Initialization strategy:
        1) All Linear layers: standard Xavier (gain=1.0) for capacity
        2) SpatialGate: initialized to near-identity (gate ≈ 1)
        3) DSU: forget gate biased to preserve previous context
        """

        # ===============================
        # 1. Default initialization for all Linear layers
        # ===============================
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1.0)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        # ===============================
        # 2. SpatialGate: near-identity initialization
        # gate(x) ≈ 1  ==>  sigmoid(bias) ≈ 1
        # ===============================
        # SpatialGate.conv = [Linear(0), GELU(1), Linear(2), Sigmoid(3)]

        # (a) First Linear: zero weight, zero bias (no dependence on input)
        nn.init.constant_(self.g.conv[0].weight, 0.0)
        nn.init.constant_(self.g.conv[0].bias, 0.0)

        # (b) Second Linear: zero weight, positive bias
        # sigmoid(4.0) ≈ 0.982  → almost identity
        nn.init.constant_(self.g.conv[2].weight, 0.0)
        nn.init.constant_(self.g.conv[2].bias, 4.0)

        # ===============================
        # 3. DSU: stabilize memory update (LSTM-style trick)
        # ===============================
        # Forget gate biased toward "keep previous c"
        nn.init.constant_(self.dsu.bf, 1.0)

        # (Optional but safe) input gate slightly conservative
        nn.init.constant_(self.dsu.bi, 0.0)
        nn.init.constant_(self.dsu.bc, 0.0)

    
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

        # text_global = text_features.mean(dim=0) / (T ** 0.5)   # [D]
        text_global = F.layer_norm(text_features.mean(dim=0), (D,))

        # ✅ Step 2: Forward Path - 逐层提取context
        # c_0 初始化为0（如图片所示）
        c_prev = torch.zeros(self.feature_dim, 
                            device=text_features.device, 
                            dtype=text_features.dtype)
        
        contexts = []  # 保存每一层的context c_l
        
        for layer_idx, proj_feat in enumerate(projected_layer_features):
            # y_l = Pool(x^l)
            y_l = proj_feat.mean(dim=0)  # [feature_dim]
            
            # c_l = DSU([y_l, text], c_{l-1})
            c_l = self.dsu(y_l, text_global, c_prev)
            
            contexts.append(c_l)
            c_prev = c_l  # 更新为下一层的输入
        
        # ===== Step3: refresh layer 23 =====
        c_23 = contexts[-2]                      # [D]
        pre_23 = projected_layer_features[-2]    # [N, D]
        refreshed_23 = self.g(pre_23, c_23)      # [N, D]

        # ===== Step4: SDPA cross-attn =====
        # Q from text
        q = self.W_q(text_features)             # [T, D]
        q = self.q_norm(q)
        print("q max:", q.max().item(), "min:", q.min().item(), "mean:", q.mean().item())

        # K from image
        k = self.W_k(refreshed_23)              # [N, D]
        k = self.k_norm(k)
        print("k max:", k.max().item(), "min:", k.min().item(), "mean:", k.mean().item())

        # V from image (你原来就是 refreshed_23)
        v = self.v_norm(refreshed_23)           # [N, D]
        print("v max:", v.max().item(), "min:", v.min().item(), "mean:", v.mean().item())

        # reshape to [B, H, L, Hd]
        H = self.num_heads
        Hd = self.head_dim

        # batch=1
        q = q.view(T, H, Hd).transpose(0, 1).unsqueeze(0)     # [1, H, T, Hd]
        k = k.view(N, H, Hd).transpose(0, 1).unsqueeze(0)     # [1, H, N, Hd]
        v = v.view(N, H, Hd).transpose(0, 1).unsqueeze(0)     # [1, H, N, Hd]

        # scaled_dot_product_attention:
        # 返回 [1, H, T, Hd]
        attn = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=0.0 if not self.training else 0.0,  # 你想加 dropout 再改
            is_causal=False
        )

        # back to [T, D]
        attn = attn.squeeze(0).transpose(1, 2).contiguous()   # [H, T, Hd] -> [H, T, Hd] then transpose -> [H, T, Hd]
        attn = attn.transpose(0, 1).contiguous().view(T, D)   # [T, D]

        output = self.W_o(attn)
        output = self.output_norm(output)

        if force_off:
            output = output * 0.0
        
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
        attn_weights_avg = attn_weights.mean(dim=0)  # [T, 24*N]
        
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