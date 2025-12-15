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
    def __init__(self, text_dim, feature_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.text_dim = text_dim
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        self.W_q = nn.Linear(text_dim, feature_dim)
        self.W_k = nn.Linear(feature_dim, feature_dim)
        self.feature_norm = LlamaRMSNorm(feature_dim)
        self.q_norm = LlamaRMSNorm(feature_dim)
        self.k_norm = LlamaRMSNorm(feature_dim)
        self.out_proj = nn.Linear(feature_dim, feature_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.W_q.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=0.1)

    def forward(self, text, features):
        """
        Args:
            text: [B, T, D_text] - text embeddings
            features: [B, N, D_feat] - image features
        Returns:
            output: [B, T, D_feat] - attended features
        """
        B, T, _ = text.shape
        _, N, _ = features.shape
        
        # ============ Step 1: Normalize features ============
        features = self.feature_norm(features)
        
        # ============ Step 2: Project to Q, K ============
        Q = self.W_q(text)       # [B, T, D_feat]
        K = self.W_k(features)   # [B, N, D_feat]
        
        # ============ Step 3: Normalize Q, K ============
        Q = self.q_norm(Q)       # [B, T, D_feat]
        K = self.k_norm(K)       # [B, N, D_feat]
        
        # ============ Step 4: Reshape for multi-head ============
        # Q: [B, T, D_feat] -> [B, T, num_heads, head_dim] -> [B, num_heads, T, head_dim]
        Q = Q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, D/H]
        K = K.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D/H]
        
        # V 直接从 features 来（不投影）
        V = features.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, N, D/H]
        
        # ============ Step 5: Compute attention ============
        attn_scores = torch.matmul(Q, K.transpose(-2, -1))  # [B, H, T, N]
        attn_scores = attn_scores / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(attn_scores, dim=-1)   # [B, H, T, N]
        
        # ============ Step 6: Apply attention to values ============
        attn_output = torch.matmul(attn_weights, V)  # [B, H, T, D/H]
        
        # ============ Step 7: Merge heads ============
        attn_output = attn_output.transpose(1, 2).contiguous()  # [B, T, H, D/H]
        attn_output = attn_output.view(B, T, self.feature_dim)  # [B, T, D_feat]
        
        # ============ Step 8: Output projection ============
        output = self.out_proj(attn_output)  # [B, T, D_feat]
        
        return output

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
        """
        目标：
        1. 三个组的总概率都接近 1/3（组间平衡）
        2. 在每个组内，选择集中到少数几层（组内集中）
        """
        # 1. 组间平衡损失
        shallow_prob = layer_probs[:, 0:8].sum(dim=-1)   # [batch_size]
        middle_prob = layer_probs[:, 8:16].sum(dim=-1)
        deep_prob = layer_probs[:, 16:24].sum(dim=-1)
        
        ideal_prob = 1.0 / 3.0
        group_balance_loss = (
            (shallow_prob - ideal_prob) ** 2 +
            (middle_prob - ideal_prob) ** 2 +
            (deep_prob - ideal_prob) ** 2
        ).mean()
        
        # 2. 组内熵损失（在每个组内鼓励集中）
        epsilon = 1e-10
        
        # 计算每个组内的条件概率分布
        shallow_probs = layer_probs[:, 0:8] / (shallow_prob.unsqueeze(-1) + epsilon)
        middle_probs = layer_probs[:, 8:16] / (middle_prob.unsqueeze(-1) + epsilon)
        deep_probs = layer_probs[:, 16:24] / (deep_prob.unsqueeze(-1) + epsilon)
        
        # 计算组内熵
        max_entropy_per_group = torch.log(torch.tensor(8.0, device=layer_probs.device))
        
        shallow_entropy = -(shallow_probs * torch.log(shallow_probs + epsilon)).sum(dim=-1)
        middle_entropy = -(middle_probs * torch.log(middle_probs + epsilon)).sum(dim=-1)
        deep_entropy = -(deep_probs * torch.log(deep_probs + epsilon)).sum(dim=-1)
        
        # 归一化到 [0, 1]
        intra_group_entropy = (
            shallow_entropy / max_entropy_per_group +
            middle_entropy / max_entropy_per_group +
            deep_entropy / max_entropy_per_group
        ).mean() / 3.0
        
        # print(f"    Balance: {group_balance_loss.item():.6f}, "
            # f"Intra-Entropy: {intra_group_entropy.item():.6f}")
        
        # 总损失：都是最小化目标
        total_loss = group_balance_loss + 0.3 * intra_group_entropy
        
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
    num_heads = config.num_heads if hasattr(config, 'num_heads') else 4
    
    return CrossAttention(text_dim=text_dim, feature_dim=feature_dim, num_heads=num_heads)

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