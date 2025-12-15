#    Copyright 2023 Haotian Liu

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector, build_cross_attn, build_layer_router

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
import torch.nn.functional as F


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        
        if hasattr(config, "mm_vision_tower"):
            # print("==========================Building vision tower================")
            self.vision_tower = build_vision_tower(config, delay_load=True)
            self.mm_projector = build_vision_projector(config)

            if getattr(config, 'use_ca', False):
                self.ca = build_cross_attn(config)
            
            if getattr(config, 'use_router', False):
                self.layer_router = build_layer_router(config)


            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def get_cross_attn(self):
        return self.ca
    
    def get_layer_router(self):
        return self.layer_router

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        self.config.use_ca = getattr(model_args, 'use_ca', False)
        self.config.use_router = getattr(model_args, 'use_router', False)

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if self.config.use_ca:
            if getattr(self, 'ca', None) is None:
                self.ca = build_cross_attn(self.config)

        if self.config.use_router:
            if getattr(self, 'layer_router', None) is None:
                self.layer_router = build_layer_router(self.config)

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)
            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')

            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))

            if self.config.use_ca and hasattr(self, 'ca') and self.ca is not None:
                ca_weights = get_w(mm_projector_weights, 'ca')
                if ca_weights:
                    self.ca.load_state_dict(ca_weights)

            if self.config.use_router and hasattr(self, 'layer_router') and self.layer_router is not None:
                router_weights = get_w(mm_projector_weights, 'layer_router')
                if router_weights:
                    self.layer_router.load_state_dict(router_weights)
        
        



def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def get_cross_attn(self):
        return self.get_model().get_cross_attn()

    def get_layer_router(self):
        return self.get_model().get_layer_router()

    def encode_images(self, images, text_token):
        """
        Batched version - supports flexible combinations of router and CA.
        Args:
            images: [batch_size, 3, H, W] 或 [batch_size, num_crops, 3, H, W]
            text_token: List[Tuple[Tensor]], 每个元素是一个 batch 的文本片段列表
        """
        # 获取24层特征
        image_features, image_forward_outs = self.get_model().get_vision_tower()(images)
        batch_size = image_features.shape[0]

        use_router = getattr(self.get_model().config, 'use_router', False)
        use_ca = getattr(self.get_model().config, 'use_ca', False)

        # Case 1: 都不使用 - 直接返回投影后的特征
        if not use_router and not use_ca:
            combined_features = self.get_model().mm_projector(image_features)
            return combined_features

        # ========== 处理文本 embeddings ==========
        # ⚠️ 关键修改：确保 text_token 的长度与 batch_size 一致
        if len(text_token) != batch_size:
            # 如果 text_token 数量少于 batch_size，用空的 tuple 填充
            while len(text_token) < batch_size:
                # 创建一个空的 tuple 作为占位符
                text_token.append(())
            # 如果 text_token 数量多于 batch_size，截断
            text_token = text_token[:batch_size]
        
        batch_text_embeddings = []
        text_lengths = []  # 记录每个 batch 的实际文本长度（不含padding）
        
        for batch_idx in range(batch_size):  # ✅ 改为使用 batch_size 而不是 len(text_token)
            # text_token[batch_idx] 是一个 tuple，包含多个文本片段
            text_segments = text_token[batch_idx]
            
            # 拼接所有文本片段
            if len(text_segments) > 0:
                batch_text = torch.cat(list(text_segments), dim=0)  # [text_len, dim]
                batch_text_embeddings.append(batch_text)
                text_lengths.append(batch_text.shape[0])
            else:
                # 如果没有文本，创建一个占位符
                # ✅ 需要确保 dummy_text 的维度正确
                if len(text_token) > 0 and len(text_token[0]) > 0:
                    dummy_dim = text_token[0][0].shape[-1]
                else:
                    # 如果完全没有文本数据，使用默认维度（通常是 LLM 的 hidden size）
                    dummy_dim = getattr(self.get_model().config, 'hidden_size', 4096)
                
                dummy_text = torch.zeros(1, dummy_dim, 
                                        device=images.device, 
                                        dtype=images.dtype if len(text_token) == 0 or len(text_token[0]) == 0 else text_token[0][0].dtype)
                batch_text_embeddings.append(dummy_text)
                text_lengths.append(0)
        
        # 找到最大长度并 pad
        max_text_len = max(t.shape[0] for t in batch_text_embeddings)
        
        padded_texts = []
        attention_masks = []  # 用于标记哪些是真实文本，哪些是padding
        
        for text_embed, actual_len in zip(batch_text_embeddings, text_lengths):
            # 创建 attention mask
            attn_mask = torch.zeros(max_text_len, device=text_embed.device, dtype=torch.bool)
            if actual_len > 0:
                attn_mask[:actual_len] = True
            
            # Pad 到 max_len
            if text_embed.shape[0] < max_text_len:
                padding = torch.zeros(
                    max_text_len - text_embed.shape[0], text_embed.shape[1],
                    device=text_embed.device, dtype=text_embed.dtype
                )
                text_embed = torch.cat([text_embed, padding], dim=0)
            
            padded_texts.append(text_embed.unsqueeze(0))  # [1, text_len, dim]
            attention_masks.append(attn_mask.unsqueeze(0))  # [1, text_len]
        
        combined_text = torch.cat(padded_texts, dim=0)  # [batch_size, text_len, dim]
        text_attention_mask = torch.cat(attention_masks, dim=0)  # [batch_size, text_len]


        # Case 2: 只使用CA，不使用Router
        if use_ca and not use_router:
            text_len = combined_text.shape[1]
            dim = combined_text.shape[-1]
            combined_features = torch.zeros(
                batch_size, text_len, dim,
                device=combined_text.device,
                dtype=combined_text.dtype
            )
            vision_hidden_states = image_forward_outs.hidden_states[1:]  # 24 layers
            
            if not self.training:
                self.get_model().ca.layer_importance_scores = []

            for layer_idx, h in enumerate(vision_hidden_states):
                # print(f"Processing layer {layer_idx + 1}/{len(vision_hidden_states)} for CA attention")
                layer_feat = h[:, 1:].to(image_features.dtype)  # [B, N, D_vit]

                layer_feat = self.get_model().mm_projector(layer_feat)  # [B, N, D_llm]

                attended = self.get_model().ca(combined_text, layer_feat)
                combined_features += attended

            if not self.training:
                self.get_model().ca.save_and_reset_attention_weights()
                
            print("conbined_features:", combined_features.shape)
            return combined_features

        # Case 3: 只使用Router，不使用CA
        if use_router and not use_ca:
            dim = self.get_model().mm_projector(image_forward_outs.hidden_states[0][:, 1:]).shape[-1]
            combined_features = torch.zeros(
                batch_size, image_features.shape[1], dim,
                device=image_features.device,
                dtype=image_features.dtype
            )
            
            if self.training:
                # 传入 attention mask 给 router
                router_output = self.get_model().layer_router(
                    combined_text, 
                    attention_mask=text_attention_mask,
                    return_loss=True
                )
                
                if len(router_output) == 4:
                    top_indices, top_weights, all_probs, diversity_loss = router_output
                    
                    if hasattr(self.get_model(), '_router_diversity_losses'):
                        self.get_model()._router_diversity_losses.append(diversity_loss)
                else:
                    top_indices, top_weights, all_probs = router_output
            else:
                top_indices, top_weights, all_probs = self.get_model().layer_router(
                    combined_text, 
                    attention_mask=text_attention_mask,
                    return_loss=False
                )
            
            # 加权融合选中的层
            for idx in range(len(top_indices)):
                layer_idx = top_indices[idx].item()
                weight = top_weights[idx]
                
                layer_feat = image_forward_outs.hidden_states[layer_idx][:, 1:].to(image_features.dtype)
                layer_features = self.get_model().mm_projector(layer_feat)
                combined_features += weight * layer_features
            
            return combined_features

        # Case 4: 同时使用Router和CA
        if use_router and use_ca:
            text_len = combined_text.shape[1]
            dim = combined_text.shape[-1]
            
            combined_features = torch.zeros(
                batch_size, text_len, dim,
                device=combined_text.device,
                dtype=combined_text.dtype
            )
            
            if self.training:
                router_output = self.get_model().layer_router(
                    combined_text, 
                    attention_mask=text_attention_mask,
                    return_loss=True
                )
                
                if len(router_output) == 4:
                    top_indices, top_weights, all_probs, diversity_loss = router_output
                    
                    if hasattr(self.get_model(), '_router_diversity_losses'):
                        self.get_model()._router_diversity_losses.append(diversity_loss)
                else:
                    top_indices, top_weights, all_probs = router_output
            else:
                top_indices, top_weights, all_probs = self.get_model().layer_router(
                    combined_text, 
                    attention_mask=text_attention_mask,
                    return_loss=False
                )
            
            # Router选中的层 + CA attention
            print("Selected layers by router:", top_indices)
            for idx in range(len(top_indices)):
                layer_idx = top_indices[idx].item()
                weight = top_weights[idx]

                layer_feat = image_forward_outs.hidden_states[layer_idx][:, 1:].to(image_features.dtype)
                layer_features = self.get_model().mm_projector(layer_feat)

                if torch.isnan(layer_features).any() or torch.isinf(layer_features).any():
                    print(f"⚠️ Layer {layer_idx}: NaN/Inf in projector output")

                attended = self.get_model().ca(combined_text, layer_features)

                if torch.isnan(attended).any() or torch.isinf(attended).any():
                    print(f"⚠️ Layer {layer_idx}: NaN/Inf in CA output")
                
                combined_features += attended

            # print("Combined features shape:", combined_features.shape)
            
            return combined_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None, CA=None
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        if self.training:
            tune_router = getattr(self.get_model().config, 'tune_router', False)
            if tune_router and hasattr(self.get_model(), 'layer_router'):
                self.get_model()._router_diversity_losses = []
        
        # 保存原始值
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        
        # 移除 padding
        input_ids_list = [cur_input_ids[cur_attention_mask] 
                        for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels_list = [cur_labels[cur_attention_mask] 
                    for cur_labels, cur_attention_mask in zip(labels, attention_mask)]
        
        # ========== 编码图像特征 ==========
        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            
            # ✅ 提取对应于concat_images的文本embeddings
            concat_batch_size = concat_images.shape[0]
            all_cur_input_embeds_no_im = []
            
            for batch_idx in range(concat_batch_size):
                # ⚠️ 确保不越界
                if batch_idx < len(input_ids_list):
                    cur_input_ids = input_ids_list[batch_idx]
                    image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
                    
                    cur_input_ids_noim = []
                    for i in range(len(image_token_indices) - 1):
                        cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                    
                    split_sizes = [x.shape[0] for x in cur_input_ids_noim]
                    
                    if len(cur_input_ids_noim) > 0 and sum(split_sizes) > 0:
                        cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
                        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
                    else:
                        cur_input_embeds_no_im = ()
                else:
                    # ✅ 如果超出范围，用空tuple填充
                    cur_input_embeds_no_im = ()
                
                all_cur_input_embeds_no_im.append(cur_input_embeds_no_im)
            
            image_features = self.encode_images(concat_images, all_cur_input_embeds_no_im)
            
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(
                                image_sizes[image_idx], self.config.image_grid_pinpoints, 
                                self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_batch_size = images.shape[0]
            all_cur_input_embeds_no_im = []
            
            for batch_idx in range(image_batch_size):
                if batch_idx < len(input_ids_list):
                    cur_input_ids = input_ids_list[batch_idx]
                    image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
                    
                    cur_input_ids_noim = []
                    for i in range(len(image_token_indices) - 1):
                        cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                    
                    split_sizes = [x.shape[0] for x in cur_input_ids_noim]
                    
                    if len(cur_input_ids_noim) > 0 and sum(split_sizes) > 0:
                        cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
                        cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
                    else:
                        cur_input_embeds_no_im = ()
                else:
                    cur_input_embeds_no_im = ()
                
                all_cur_input_embeds_no_im.append(cur_input_embeds_no_im)
            
            image_features = self.encode_images(images, all_cur_input_embeds_no_im)

        # ========== 合并文本和图像 embeddings ==========
        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        
        for batch_idx, cur_input_ids in enumerate(input_ids_list):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels_list[batch_idx])
                cur_image_idx += 1
                continue

            # ✅ 重新提取当前batch的文本embeddings（避免使用之前可能填充的空tuple）
            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            
            cur_input_ids_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
            
            split_sizes = [x.shape[0] for x in cur_input_ids_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            
            # 获取对应的 labels
            cur_labels = labels_list[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])

            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, 
                                                    device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Padding
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, 
                                    dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), 
                                    dtype=_attention_mask.dtype if _attention_mask is not None else torch.bool, 
                                    device=input_ids.device)
        position_ids = torch.zeros((batch_size, max_len), 
                                dtype=_position_ids.dtype if _position_ids is not None else torch.long, 
                                device=input_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), 
                            dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, 
                                                            dtype=position_ids.dtype, 
                                                            device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), 
                            dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, 
                                                            dtype=position_ids.dtype, 
                                                            device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None

        if _position_ids is None:
            position_ids = None

        # print("Prepared input embeds shape:", new_input_embeds.shape)

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False