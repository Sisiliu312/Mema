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

class AttentionWeightSaver:
    def __init__(self, save_dir='attention_weights', format='pt'):
        """
        初始化保存器
        :param save_dir: 保存目录路径
        :param format: 保存格式 ('pt' for PyTorch, 'npy' for NumPy)
        """
        self.save_dir = save_dir
        self.format = format
        self.counter = 0
        self._create_save_dir()
        
    def _create_save_dir(self):
        """创建保存目录"""
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
            
    def _generate_filename(self):
        """生成按序号递增的文件名"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"attn_{self.counter:04d}_{timestamp}"
        self.counter += 1
        return os.path.join(self.save_dir, f"{filename}.{self.format}")
    
    def save(self, attn_weights, metadata=None):
        """
        保存attention weights
        :param attn_weights: 要保存的attention weights张量
        :param metadata: 可选的元数据字典
        """
        filename = self._generate_filename()
        
        if self.format == 'pt':
            # 保存为PyTorch文件
            save_dict = {'attn_weights': attn_weights}
            if metadata:
                save_dict['metadata'] = metadata
            torch.save(save_dict, filename)
        elif self.format == 'npy':
            # 保存为NumPy文件
            if isinstance(attn_weights, torch.Tensor):
                attn_weights = attn_weights.cpu().numpy()
            if metadata:
                np.savez(filename, attn_weights=attn_weights, **metadata)
            else:
                np.save(filename, attn_weights)
        else:
            raise ValueError(f"Unsupported format: {self.format}")
            
        print(f"Saved attention weights to: {filename}")
        return filename
    
saver = AttentionWeightSaver(save_dir='/home/data/shika/LLaVA/playground/data/eval/textvqa', format='pt')

class CrossAttention(nn.Module):
    def __init__(self, text_dim, feature_dim):
        super(CrossAttention, self).__init__()
        self.text_dim = text_dim
        self.feature_dim = feature_dim
        self.W_q = nn.Linear(text_dim, feature_dim)
        self.W_k = nn.Linear(feature_dim, feature_dim)
        # self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_norm = LlamaRMSNorm(feature_dim)
        # self.output_norm = LlamaRMSNorm(feature_dim)
        self.q_norm = LlamaRMSNorm(feature_dim)
        self.k_norm = LlamaRMSNorm(feature_dim)
        
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)

    def forward(self, text, features):
        # ============ 步骤1: 检查输入 ============
        # print(f"\n{'='*60}")
        # print("🔍 CrossAttention Debug:")
        # print(f"  text shape: {text.shape}, range: [{text.min():.4f}, {text.max():.4f}]")
        # print(f"  features shape: {features.shape}, range: [{features.min():.4f}, {features.max():.4f}]")
        
        # ============ 步骤2: 线性变换 ============
        features = self.feature_norm(features)
        Q = self.W_q(text)
        K = self.W_k(features)
        Q = self.q_norm(Q)
        K = self.k_norm(K)
        
        # print(f"  After linear transform:")
        # print(f"    Q range: [{Q.min():.4f}, {Q.max():.4f}], has_nan={torch.isnan(Q).any()}, has_inf={torch.isinf(Q).any()}")
        # print(f"    K range: [{K.min():.4f}, {K.max():.4f}], has_nan={torch.isnan(K).any()}, has_inf={torch.isinf(K).any()}")
        
        # ✅ 检查权重
        # print(f"  Weight stats:")
        # print(f"    W_q.weight: range=[{self.W_q.weight.min():.4f}, {self.W_q.weight.max():.4f}], norm={self.W_q.weight.norm():.4f}")
        # print(f"    W_k.weight: range=[{self.W_k.weight.min():.4f}, {self.W_k.weight.max():.4f}], norm={self.W_k.weight.norm():.4f}")
        
        # ============ 如果Q或K已经有Inf，直接返回零 ============
        # if torch.isinf(Q).any() or torch.isinf(K).any():
        #     print("⚠️ Q or K contains Inf! Returning zeros to avoid crash.")
        #     return torch.zeros_like(text)
        
        # ============ 步骤3: 计算attention scores ============
        attn_scores = torch.matmul(Q, K.transpose(1, 2))
        
        # if torch.isnan(attn_scores).any() or torch.isinf(attn_scores).any():
        #     print(f"⚠️ attn_scores after matmul: has_nan={torch.isnan(attn_scores).any()}, has_inf={torch.isinf(attn_scores).any()}")
        #     print(f"   attn_scores range: [{attn_scores.min():.4f}, {attn_scores.max():.4f}]")
        
        attn_scores = attn_scores / (K.size(-1) ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        attended = torch.matmul(attn_weights, features)
        # attended = self.output_norm(attended)
        
        # print(f"{'='*60}\n")
        
        return attended

class AttentionLayerRouter(nn.Module):
    def __init__(self, dim, num_layers, top_router):
        super(AttentionLayerRouter, self).__init__()
        self.num_layers = num_layers
        self.top_router = top_router
        self.dim = dim
        
        # Attention pooling for text tokens
        self.attention_pool = nn.Sequential(
            nn.Linear(dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Linear(256, num_layers)
        )
        
        self._reset_parameters()
    
    def _reset_parameters(self):
        for m in self.router.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

        nn.init.orthogonal_(self.router[0].weight, gain=0.5)
        nn.init.constant_(self.router[0].bias, 0.0)
        
        with torch.no_grad():
            self.router[2].weight.normal_(0, 0.0001)
            uniform_indices = [22]
            for idx in uniform_indices:
                self.router[2].bias[idx] = 1
    
    def compute_diversity_loss(self, layer_probs):
        """修正版多样性损失"""
        # 1. 组间平衡损失
        shallow_prob = layer_probs[:, 0:8].sum(dim=-1)
        middle_prob = layer_probs[:, 8:16].sum(dim=-1)
        deep_prob = layer_probs[:, 16:24].sum(dim=-1)
        
        ideal_prob = 1.0 / 3.0
        group_balance_loss = (
            (shallow_prob - ideal_prob) ** 2 +
            (middle_prob - ideal_prob) ** 2 +
            (deep_prob - ideal_prob) ** 2
        ).mean()
        
        # 2. 熵损失
        epsilon = 1e-10
        entropy = -(layer_probs * torch.log(layer_probs + epsilon)).sum(dim=-1)
        max_entropy = torch.log(
            torch.tensor(float(self.num_layers), 
                        device=layer_probs.device, 
                        dtype=layer_probs.dtype)
        )
        
        normalized_entropy = 1.0 - (entropy / max_entropy)
        entropy_loss = normalized_entropy.mean()
        
        total_loss = group_balance_loss + 0.3 * entropy_loss
        
        return total_loss

    def forward(self, text_features, attention_mask=None, return_loss=False):
        """
        Args:
            text_features: [batch_size, text_len, dim]
            attention_mask: [batch_size, text_len] bool tensor, True for valid tokens
            return_loss: 是否返回多样性损失
        """
        batch_size, text_len, dim = text_features.shape
        
        # ✅ Attention pooling with mask
        attn_weights = self.attention_pool(text_features)  # [batch_size, text_len, 1]
        
        if attention_mask is not None:
            # 将 padding 位置的权重设为极小值
            attn_weights = attn_weights.masked_fill(
                ~attention_mask.unsqueeze(-1), 
                float('-inf')
            )
        
        attn_weights = F.softmax(attn_weights, dim=1)  # [batch_size, text_len, 1]
        
        # 加权求和
        pooled_features = (text_features * attn_weights).sum(dim=1)  # [batch_size, dim]
        pooled_features = F.normalize(pooled_features, p=2, dim=-1) * (self.dim ** 0.5)
        
        # Router 预测
        logits = self.router(pooled_features)  # [batch_size, num_layers]
        temperature = 2.0
        layer_probs = F.softmax(logits / temperature, dim=-1)

        unified_probs = layer_probs.mean(dim=0)  # [num_layers]
        
        top_weights, top_indices = torch.topk(unified_probs, self.top_router)
        top_weights = top_weights / top_weights.sum()
        
        
        if return_loss:
            diversity_loss = self.compute_diversity_loss(layer_probs)
            return top_indices, top_weights, unified_probs, diversity_loss
        
        return top_indices, top_weights, unified_probs

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

def build_cross_attn(config):
    text_dim = config.text_dim if hasattr(config, 'text_dim') else 4096
    feature_dim = config.feature_dim if hasattr(config, 'feature_dim') else 4096
    
    return CrossAttention(text_dim=text_dim, feature_dim=feature_dim)

def build_layer_router(config):
    # print("=" * 60)
    # print("🔍 build_layer_router 调试:")
    # print(f"  hasattr(config, 'dim'): {hasattr(config, 'dim')}")
    # print(f"  hasattr(config, 'num_layers'): {hasattr(config, 'num_layers')}")
    # print(f"  hasattr(config, 'top_router'): {hasattr(config, 'top_router')}")
    
    dim = config.dim if hasattr(config, 'dim') else 4096
    num_layers = config.num_layers if hasattr(config, 'num_layers') else 24
    top_router = config.top_router if hasattr(config, 'top_router') else 5
    
    # print(f"  最终: dim={dim}, num_layers={num_layers}, top_router={top_router}")
    # print(f"  类型: dim={type(dim)}, num_layers={type(num_layers)}, top_router={type(top_router)}")
    # print("=" * 60)
    
    return AttentionLayerRouter(dim=dim, num_layers=num_layers, top_router=top_router)