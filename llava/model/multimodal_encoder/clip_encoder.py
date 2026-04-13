import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from .clip import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig


import logging


logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

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
    """单 memory [N, D]，与 dynamic-image-mean-max 一致。"""
    def __init__(self, dim, reduction_ratio=4):
        super().__init__()
        self.dim = dim
        self.reduced_dim = max(dim // reduction_ratio, 1)
        self.W1 = nn.Linear(3 * dim, self.reduced_dim)
        self.Wc = nn.Linear(self.reduced_dim, dim)
        self.Wi = nn.Linear(self.reduced_dim, dim)
        self.Wf = nn.Linear(self.reduced_dim, dim)
        self.bc = nn.Parameter(torch.zeros(dim))
        self.bi = nn.Parameter(torch.zeros(dim))
        self.bf = nn.Parameter(torch.zeros(dim))
        self._reset_parameters()
        # print(f"DSU reduction ratio: {reduction_ratio}")
        # print(f"DSU dimension: {dim}")
        # print(f"DSU reduced dimension: {self.reduced_dim}")

    def _reset_parameters(self):
        nn.init.constant_(self.bf, 1.0)
        nn.init.constant_(self.bi, 0.0)
        nn.init.constant_(self.bc, 0.0)

    def forward(self, y_l, text_global, c_prev):
        """y_l/text_global: [N, D]；c_prev: [N, D]。返回 c_l [N, D]。"""
        c_prev_norm = F.layer_norm(c_prev, (c_prev.shape[-1],))
        combined = torch.cat([c_prev_norm, y_l, text_global], dim=-1)
        s = F.relu(self.W1(combined))
        c_tilde = torch.tanh(self.Wc(s) + self.bc)
        i = torch.sigmoid(self.Wi(s) + self.bi)
        f = torch.sigmoid(self.Wf(s) + self.bf)
        c_l = f * c_prev + i * c_tilde
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
        self.dsu_reduction_ratio = int(getattr(args, 'dsu_reduction_ratio', 4))

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
            self.y_proj = nn.Linear(3 * h, h)
            self.dsu = DynamicSharingUnit(dim=h, reduction_ratio=self.dsu_reduction_ratio)
            self.spatial_gate = SpatialGate(h)

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

        # 若 delay_load 时已创建且已从 checkpoint 加载，不要用新初始化覆盖（避免覆盖 dsu/spatial_gate/text_global_proj/y_proj）
        if getattr(self, 'dsu', None) is not None:
            if self.text_global_proj is not None:
                self.text_global_proj = self.text_global_proj.to(device=dev, dtype=dtype)
            if getattr(self, 'y_proj', None) is not None:
                self.y_proj = self.y_proj.to(device=dev, dtype=dtype)
            self.dsu = self.dsu.to(device=dev, dtype=dtype)
            self.spatial_gate = self.spatial_gate.to(device=dev, dtype=dtype)
        else:
            self.text_global_proj = nn.Linear(self.llm_hidden_size, h).to(device=dev, dtype=dtype)
            self.y_proj = nn.Linear(3 * h, h).to(device=dev, dtype=dtype)
            self.dsu = DynamicSharingUnit(dim=h, reduction_ratio=self.dsu_reduction_ratio).to(device=dev, dtype=dtype)
            self.spatial_gate = SpatialGate(h).to(device=dev, dtype=dtype)
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

        # 单 memory [N, D]，与 dynamic-image-mean-max 一致
        c_prev = torch.zeros(N, D, device=dev, dtype=dtype)

        encoder_states = (hidden_states,)
        all_attentions = () if output_attentions else None

        for layer_idx, encoder_layer in enumerate(vm.encoder.layers):
            layer_outputs = encoder_layer(
                hidden_states,
                None,
                None,
                output_attentions=output_attentions,
            )
            hidden_states = layer_outputs[0]
            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

            # 多视角摘要 mean+max+cls 压回 D
            mean = hidden_states.mean(dim=1)
            maxv = hidden_states.amax(dim=1)
            cls_tok = hidden_states[:, 0, :]
            y_l = self.y_proj(torch.cat([mean, maxv, cls_tok], dim=-1))
            c_l = self.dsu(y_l, text_global_expanded, c_prev)
            c_prev = c_l
            flat = hidden_states.reshape(N * L, D)
            c_l_flat = c_l.unsqueeze(1).expand(-1, L, -1).reshape(N * L, D)
            refreshed = self.spatial_gate(flat, c_l_flat).reshape(N, L, D)
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
        return image_forward_outs, c_prev

    def forward(self, images, text_global=None, image_split_sizes=None, answer_global=None, answer_valid_mask=None):
        """返回 (image_features, c_sem)。c_sem 即单 memory c_24，供 mm_proj 后与 LLM last hidden 对齐。"""
        output_dir = "attention_maps/attention_layer_image1"
        os.makedirs(output_dir, exist_ok=True)

        text_global_proj = None
        c_sem = None
        if text_global is not None and self.text_global_proj is not None:
            text_global_proj = self.text_global_proj(text_global.to(device=self.device, dtype=self.dtype))
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
                image_forward_outs = image_forward_out
        else:
            if text_global_proj is not None and getattr(self, 'dsu', None) is not None:
                image_forward_outs, c = self._forward_vision_with_dsu(
                    images.to(device=self.device, dtype=self.dtype),
                    text_global_proj,
                    image_split_sizes=image_split_sizes,
                    output_attentions=True,
                )
                c_sem = c  # 单 memory c_24，供 mm_proj 后与 LLM hidden 对齐
            else:
                with torch.no_grad():
                    image_forward_outs = self.vision_tower(
                        images.to(device=self.device, dtype=self.dtype),
                        output_hidden_states=True,
                        output_attentions=True,
                    )
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features, c_sem

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
