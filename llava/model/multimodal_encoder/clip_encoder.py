import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .clip import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


import logging


logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)


# ----- 与 projector builder 一致的 DSU + SpatialGate，用于在 CLIP 每层后做 LSTM 积累并用 c_l refresh -----
# class SpatialGate(nn.Module):
#     """乘法 gate：gate = conv(x + c_l)，输出 x * gate。"""
#     def __init__(self, feature_dim):
#         super().__init__()
#         self.conv = nn.Sequential(
#             nn.Linear(feature_dim, feature_dim),
#             nn.GELU(),
#             nn.Linear(feature_dim, feature_dim),
#             nn.Sigmoid()
#         )
#         self._reset_parameters()

#     def _reset_parameters(self):
#         """近恒等初始化：让 gate 初始时接近 1，使 x * gate ≈ x。"""
#         nn.init.constant_(self.conv[0].weight, 0.0)
#         nn.init.constant_(self.conv[0].bias, 0.0)
#         nn.init.constant_(self.conv[2].weight, 0.0)
#         # 初始化 bias 为较大正数，使 sigmoid(bias) ≈ 1
#         nn.init.constant_(self.conv[2].bias, 5.0)

#     def forward(self, x, c_l):
#         """x: [N*L, D] 或 [N, D]；c_l: [N*L, D] 或 [N, D]（与 x 同形状）"""
#         if c_l.dim() == 1:
#             c = c_l.unsqueeze(0).expand_as(x)
#         else:
#             c = c_l
#         gate = self.conv(x + c)
#         return x * gate



class SpatialGate(nn.Module):
    def __init__(self, feature_dim, num_layers=24):
        super().__init__()
        self.num_layers = num_layers
        self.delta = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )
        # token-wise alpha：由 h 预测每个 token 的 alpha
        self.alpha_proj = nn.Linear(feature_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self):
        """近恒等初始化：delta≈0，token_alpha 初始较小，使 gate 初始时几乎不改变 x。"""
        nn.init.constant_(self.delta[0].weight, 0.0)
        nn.init.constant_(self.delta[0].bias, 0.0)
        nn.init.constant_(self.delta[2].weight, 0.0)
        nn.init.constant_(self.delta[2].bias, 0.0)
        nn.init.zeros_(self.alpha_proj.weight)
        nn.init.constant_(self.alpha_proj.bias, -2.2)  # sigmoid(-2.2) ≈ 0.1

    # def forward(self, x, c_l):
    #     """x: [N, D]; c_l: [D] 或 [N, D]（与 x 的 batch 对应，若 2D 则需与 x 同长度）"""
    #     if c_l.dim() == 1:
    #         c = c_l.unsqueeze(0).expand_as(x)
    #     else:
    #         c = c_l
    #     h = F.layer_norm(x + c, (x.size(-1),))
    #     delta = torch.tanh(self.delta(h))
    #     a = torch.tanh(self.alpha)
    #     return x + a * delta

    
    def forward(self, x, c_l, layer_idx=None, prefix=""):
        # x: [N, D]
        if c_l.dim() == 1:
            c = c_l.unsqueeze(0).expand_as(x)
        else:
            c = c_l

        h = F.layer_norm(x + c, (x.size(-1),))

        # 1) 看 tanh 之前的 delta 幅度（是否过大导致 tanh 饱和）
        delta_pre = self.delta(h)                      # [N, D]
        delta = torch.tanh(delta_pre)                 # [N, D]

        # token-wise alpha
        token_alpha = torch.sigmoid(self.alpha_proj(h))  # [N, 1]

        update = token_alpha * delta
        x_ref = x + update

        # eps = 1e-8
        # x_norm = x.norm(dim=-1) + eps                 # [N]
        # upd_norm = update.norm(dim=-1)                # [N]
        # ratio = (upd_norm / x_norm).mean()

        # # 2) 方向指标：cos(x, update) 以及 cos(x, x_ref)
        # cos_x_upd = F.cosine_similarity(x, update, dim=-1).mean()
        # cos_x_xref = F.cosine_similarity(x, x_ref, dim=-1).mean()

        # # 3) token 级别：更新是不是集中在少数 token？
        # #    这里 x 是 [N,D]（你外面把 N=token 扁平了），所以直接按 N 排序即可
        # r = (upd_norm / x_norm).detach()              # [N]
        # r_sorted, _ = torch.sort(r)
        # top1 = r_sorted[int(0.99 * (r_sorted.numel()-1))]
        # top10 = r_sorted[int(0.90 * (r_sorted.numel()-1))]
        # med = r_sorted[int(0.50 * (r_sorted.numel()-1))]

        # # 4) 饱和比例：delta 有多少维接近 ±1（tanh 饱和）
        # sat = (delta.abs() > 0.98).float().mean()
        # # 也可以看 pre-tanh 的幅度分布
        # pre_abs_mean = delta_pre.abs().mean()
        # pre_abs_max = delta_pre.abs().max()

        # # ---- print (token-wise alpha: [N,1]，打印 min/max/mean) ----
        # tag = f"{prefix}[Gate"
        # if layer_idx is not None:
        #     tag += f" L{layer_idx}"
        # tag += "]"
        # ta = token_alpha.squeeze(-1)  # [N]
        # print(
        #     f"{tag} "
        #     f"token_alpha min={ta.min().item():.4f} max={ta.max().item():.4f} mean={ta.mean().item():.4f} | "
        #     f"ratio={ratio.item():.4f} | "
        #     f"cos(x,upd)={cos_x_upd.item():.4f} cos(x,x_ref)={cos_x_xref.item():.4f} | "
        #     f"r_med={med.item():.4f} r_top10%={top10.item():.4f} r_top1%={top1.item():.4f} | "
        #     f"sat(|tanh|>0.98)={sat.item():.3f} | "
        #     f"|pre|mean={pre_abs_mean.item():.3f} |pre|max={pre_abs_max.item():.3f}"
        # )

        return x_ref


class DynamicSharingUnit(nn.Module):
    """双通道记忆：c^tex（纹理/局部）与 c^sem（语义/全局），按深度日程混合更新。"""
    def __init__(self, dim, reduction_ratio=4):
        super().__init__()
        assert dim % 2 == 0, "dim 需为偶数以便 chunk(2)"
        self.dim = dim
        self.dim_half = dim // 2
        self.reduced_dim = max(self.dim_half // reduction_ratio, 1)
        # tex 通道：偏局部、纹理
        self.W1_tex = nn.Linear(3 * self.dim_half, self.reduced_dim)
        self.Wc_tex = nn.Linear(self.reduced_dim, self.dim_half)
        self.Wi_tex = nn.Linear(self.reduced_dim, self.dim_half)
        self.Wf_tex = nn.Linear(self.reduced_dim, self.dim_half)
        self.bc_tex = nn.Parameter(torch.zeros(self.dim_half))
        self.bi_tex = nn.Parameter(torch.zeros(self.dim_half))
        self.bf_tex = nn.Parameter(torch.zeros(self.dim_half))
        # sem 通道：偏全局、语义
        self.W1_sem = nn.Linear(3 * self.dim_half, self.reduced_dim)
        self.Wc_sem = nn.Linear(self.reduced_dim, self.dim_half)
        self.Wi_sem = nn.Linear(self.reduced_dim, self.dim_half)
        self.Wf_sem = nn.Linear(self.reduced_dim, self.dim_half)
        self.bc_sem = nn.Parameter(torch.zeros(self.dim_half))
        self.bi_sem = nn.Parameter(torch.zeros(self.dim_half))
        self.bf_sem = nn.Parameter(torch.zeros(self.dim_half))
        self._reset_parameters()

    def _reset_parameters(self):
        """LSTM 风格：forget gate 偏置 1，input gate 偏置 0。"""
        for name in ("tex", "sem"):
            nn.init.constant_(getattr(self, f"bf_{name}"), 1.0)
            nn.init.constant_(getattr(self, f"bi_{name}"), 0.0)
            nn.init.constant_(getattr(self, f"bc_{name}"), 0.0)

    def forward(self, y_l, text_global, c_prev, layer_idx, num_layers):
        """layer_idx: 当前层 0..L-1；num_layers: 总层数。浅层更更新 tex，深层更更新 sem。"""
        c_prev_tex, c_prev_sem = c_prev.chunk(2, dim=-1)
        y_tex, y_sem = y_l.chunk(2, dim=-1)
        t_tex, t_sem = text_global.chunk(2, dim=-1)

        w = layer_idx / max(num_layers - 1, 1)  # 0 -> 1，深层大

        # tex 流
        c_prev_tex_norm = F.layer_norm(c_prev_tex, (c_prev_tex.shape[-1],))
        combined_tex = torch.cat([c_prev_tex_norm, y_tex, t_tex], dim=-1)
        s_tex = F.relu(self.W1_tex(combined_tex))
        c_tilde_tex = torch.tanh(self.Wc_tex(s_tex) + self.bc_tex)
        i_tex = torch.sigmoid(self.Wi_tex(s_tex) + self.bi_tex)
        f_tex = torch.sigmoid(self.Wf_tex(s_tex) + self.bf_tex)
        c_tex = f_tex * c_prev_tex + (1.0 - w) * i_tex * c_tilde_tex

        # sem 流
        c_prev_sem_norm = F.layer_norm(c_prev_sem, (c_prev_sem.shape[-1],))
        combined_sem = torch.cat([c_prev_sem_norm, y_sem, t_sem], dim=-1)
        s_sem = F.relu(self.W1_sem(combined_sem))
        c_tilde_sem = torch.tanh(self.Wc_sem(s_sem) + self.bc_sem)
        i_sem = torch.sigmoid(self.Wi_sem(s_sem) + self.bi_sem)
        f_sem = torch.sigmoid(self.Wf_sem(s_sem) + self.bf_sem)
        c_sem = f_sem * c_prev_sem + w * i_sem * c_tilde_sem

        c_l = torch.cat([c_tex, c_sem], dim=-1)
        # print("c_l max", c_l.max(),"c_l min", c_l.min(),"c_l mean", c_l.mean())
        return c_l


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        # LLM hidden_size (e.g. 4096)，用于将 text_global 投影到 CLIP hidden (1024)
        self.llm_hidden_size = getattr(args, 'hidden_size', 4096)
        self.text_global_proj = None  # 在 load_model 中创建：Linear(4096, 1024)

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        # self.select_layer = -1
        # print("-----------------------------------",self.select_layer)
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)
            # delay_load 时也创建 text_global 投影，以便后续 forward 可用
            self.text_global_proj = nn.Linear(self.llm_hidden_size, self.cfg_only.hidden_size)
            h = self.cfg_only.hidden_size
            num_layers = getattr(self.cfg_only, 'num_hidden_layers', 24)
            self.dsu = DynamicSharingUnit(dim=h, reduction_ratio=4)
            self.spatial_gate = SpatialGate(h, num_layers=num_layers)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        dev = self.vision_tower.device
        dtype = self.vision_tower.dtype
        h = self.vision_tower.config.hidden_size

        # 若 delay_load 时已创建且已从 checkpoint 加载，不要用新初始化覆盖（避免覆盖 dsu/spatial_gate/text_global_proj）
        if getattr(self, 'dsu', None) is not None:
            if self.text_global_proj is not None:
                self.text_global_proj = self.text_global_proj.to(device=dev, dtype=dtype)
            self.dsu = self.dsu.to(device=dev, dtype=dtype)
            self.spatial_gate = self.spatial_gate.to(device=dev, dtype=dtype)
        else:
            num_layers = len(self.vision_tower.vision_model.encoder.layers)
            self.text_global_proj = nn.Linear(self.llm_hidden_size, h).to(device=dev, dtype=dtype)
            self.dsu = DynamicSharingUnit(dim=h, reduction_ratio=4).to(device=dev, dtype=dtype)
            self.spatial_gate = SpatialGate(h, num_layers=num_layers).to(device=dev, dtype=dtype)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    def _forward_vision_with_dsu(self, pixel_values, text_global_proj, image_split_sizes=None, output_attentions=True):
        """逐层跑 CLIP encoder，每层后用 DSU 积累 c_l，用 SpatialGate 以 c_l 做 refresh，再输入下一层。
        text_global_proj [B, D] 每个样本一个；image_split_sizes 为每样本图像数，用于把 text 对齐到 N 张图。
        """
        vm = self.vision_tower.vision_model
        dev = pixel_values.device
        dtype = pixel_values.dtype
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)

        hidden_states = vm.embeddings(pixel_values)
        hidden_states = vm.pre_layrnorm(hidden_states)
        N, L, D = hidden_states.shape

        # 每个样本有自己的 text_global [B, D]；按 image 数展开为 [N, D]，与 N 张图一一对应
        if image_split_sizes is not None and len(image_split_sizes) == text_global_proj.shape[0]:
            text_global_expanded = torch.repeat_interleave(
                text_global_proj.to(device=dev, dtype=dtype),
                torch.tensor(image_split_sizes, device=dev, dtype=torch.long),
                dim=0,
            )
        else:
            text_global_expanded = text_global_proj.to(device=dev, dtype=dtype)
        assert text_global_expanded.shape[0] == N, "text_global 展开后应与图像数 N 一致"

        # c_prev 带 batch 维 [N, D]：N 为当前 batch 的图像数，每张图一个 context 向量
        c_prev = torch.zeros(N, D, device=dev, dtype=dtype)

        encoder_states = (hidden_states,)
        all_attentions = () if output_attentions else None
        num_layers = len(vm.encoder.layers)

        # 逐层：第 l 层用本层算出的 c_l 做 gate refresh，refresh 后的特征再输入第 l+1 层（c_1..c_24 每层一个）
        for layer_idx, encoder_layer in enumerate(vm.encoder.layers):
            # print("第i层:", layer_idx)
            layer_outputs = encoder_layer(
                hidden_states,
                None,
                None,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            # 本层 l：y_l = Pool(本层输出)，c_l = DSU 双通道(tex/sem) + 深度日程
            y_l = hidden_states.mean(dim=1)
            c_l = self.dsu(y_l, text_global_expanded, c_prev, layer_idx=layer_idx, num_layers=num_layers)
            c_prev = c_l
            # 用本层的 c_l 做 gate refresh，refresh 后的结果作为下一层的输入
            flat = hidden_states.reshape(N * L, D)
            c_l_flat = c_l.unsqueeze(1).expand(-1, L, -1).reshape(N * L, D)
            refreshed = self.spatial_gate(flat, c_l_flat, layer_idx=layer_idx).reshape(N, L, D)
            hidden_states = refreshed

            encoder_states = encoder_states + (hidden_states,)

        last_hidden_state = hidden_states
        pooled_output = last_hidden_state[:, 0, :]
        pooled_output = vm.post_layernorm(pooled_output)

        from transformers.modeling_outputs import BaseModelOutputWithPooling
        image_forward_outs = BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_states,
            attentions=all_attentions,
        )
        # 返回最后一层 c_l (c_prev) 供 L_align 使用
        return image_forward_outs, c_prev

    def forward(self, images, text_global=None, image_split_sizes=None, answer_global=None, answer_valid_mask=None):
        output_dir = "attention_maps/attention_layer_image1"
        os.makedirs(output_dir, exist_ok=True)

        text_global_proj = None
        if text_global is not None and self.text_global_proj is not None:
            # print("text_global shape", text_global.shape)
            text_global_proj = self.text_global_proj(text_global.to(device=self.device, dtype=self.dtype))
            align_loss = None
        if type(images) is list:
            with torch.no_grad():
                image_features = []
                for idx, image in enumerate(images):
                    image_forward_out = self.vision_tower(
                        image.to(device=self.device, dtype=self.dtype).unsqueeze(0),
                        output_hidden_states=True,
                        output_attentions=True,
                    )
                    image_feature = self.feature_select(image_forward_out).to(image.dtype)
                    image_features.append(image_feature)
                    self._save_attention_maps(image_forward_out.attentions, idx, output_dir)
                image_forward_outs = image_forward_out
                align_loss = None
        else:
            if text_global_proj is not None and getattr(self, 'dsu', None) is not None:
                image_forward_outs, c_L = self._forward_vision_with_dsu(
                    images.to(device=self.device, dtype=self.dtype),
                    text_global_proj,
                    image_split_sizes=image_split_sizes,
                    output_attentions=True,
                )
                # auxiliary loss: L_align = 1 - cosine_similarity(c_L, a)
                if answer_global is not None and self.text_global_proj is not None and answer_valid_mask is not None and answer_valid_mask.any():
                    a = self.text_global_proj(answer_global.to(device=self.device, dtype=self.dtype))  # [B, D]
                    if image_split_sizes is not None:
                        c_L_list = torch.split(c_L, image_split_sizes, dim=0)
                        c_L_agg = torch.stack([x.mean(dim=0) for x in c_L_list], dim=0)  # [B, D]
                    else:
                        c_L_agg = c_L
                    cos = F.cosine_similarity(c_L_agg, a, dim=-1)
                    align_loss = (1 - cos)[answer_valid_mask].mean()
            else:
                with torch.no_grad():
                    image_forward_outs = self.vision_tower(
                        images.to(device=self.device, dtype=self.dtype),
                        output_hidden_states=True,
                        output_attentions=True,
                    )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
            self._save_attention_maps(image_forward_outs.attentions, 0, output_dir)

        return image_features, image_forward_outs, align_loss

    def _save_attention_maps(self, attentions, image_idx, output_dir):
        """
        只保存 CLS Token 的 Attention Map，输出形状为 (seq_len,)
        """
        for layer_idx, attention in enumerate(attentions):
            # attention shape: (batch_size=1, num_heads, seq_len, seq_len)
            attention_map = attention.mean(dim=1)  
            # 对 num_heads 取平均，形状变为 (1, seq_len, seq_len)
            cls_attention = attention_map[0, :, :]  
            # 保存 CLS Attention Map
            file_path = os.path.join(output_dir, f"image_{image_idx}_layer_{layer_idx}_cls.pt")
            torch.save(cls_attention, file_path)

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self.vision_tower.requires_grad_(False)

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True


    @torch.no_grad()
    def forward_feature(self, images):

        image_forward_outs = self.vision_tower(
            images.to(device=self.device, dtype=self.dtype),
            output_hidden_states=True,
            output_attentions=True  # 启用 Attention Map 输出
        )
        
        image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features, image_forward_outs

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
