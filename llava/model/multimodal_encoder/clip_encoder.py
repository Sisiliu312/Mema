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
        # Token-wise alpha: predict a gate weight for each token from h
        self.alpha_proj = nn.Linear(feature_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self):
        """Near-identity initialization: delta≈0 and token_alpha starts small so the gate barely changes x at initialization."""
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

        # 1) Inspect delta magnitude before tanh (too large may cause tanh saturation)
        delta_pre = self.delta(h)                      # [N, D]
        delta = torch.tanh(delta_pre)                 # [N, D]

        # Token-wise alpha
        token_alpha = torch.sigmoid(self.alpha_proj(h))  # [N, 1]

        update = token_alpha * delta
        x_ref = x + update

        return x_ref


class DynamicSharingUnit(nn.Module):
    """Single memory [N, D], consistent with dynamic-image-mean-max."""
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
        # print(f"DSU original dimension: {dim}")
        # print(f"DSU reduced dimension: {self.reduced_dim}")

    def _reset_parameters(self):
        nn.init.constant_(self.bf, 1.0)
        nn.init.constant_(self.bi, 0.0)
        nn.init.constant_(self.bc, 0.0)

    def forward(self, y_l, text_global, c_prev):
        """y_l/text_global: [N, D]; c_prev: [N, D]. Returns c_l [N, D]."""
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
        # LLM hidden size (e.g., 4096), used to project text_global to CLIP hidden size (e.g., 1024)
        self.llm_hidden_size = getattr(args, 'hidden_size', 4096)
        self.text_global_proj = None  # Created in load_model: Linear(4096, 1024)

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
            # Also create text_global projection in delay_load mode so later forward calls can use it
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

        # If modules were already created during delay_load and restored from checkpoint,
        # do not overwrite them with fresh initialization
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
        """Run CLIP encoder layer by layer: after each layer, accumulate c_l with DSU,
        refresh hidden states with SpatialGate conditioned on c_l, then feed into next layer.
        text_global_proj [B, D] has one vector per sample; image_split_sizes is image count per sample,
        used to align text features to N images.
        """
        vm = self.vision_tower.vision_model
        dev = pixel_values.device
        dtype = pixel_values.dtype
        pixel_values = pixel_values.to(device=self.device, dtype=self.dtype)

        hidden_states = vm.embeddings(pixel_values)
        hidden_states = vm.pre_layrnorm(hidden_states)
        N, L, D = hidden_states.shape

        # Each sample has its own text_global [B, D]; expand by image count to [N, D] to match N images
        if image_split_sizes is not None and len(image_split_sizes) == text_global_proj.shape[0]:
            text_global_expanded = torch.repeat_interleave(
                text_global_proj.to(device=dev, dtype=dtype),
                torch.tensor(image_split_sizes, device=dev, dtype=torch.long),
                dim=0,
            )
        else:
            text_global_expanded = text_global_proj.to(device=dev, dtype=dtype)
        assert text_global_expanded.shape[0] == N, "Expanded text_global must match image count N"

        # Single memory vector [N, D], consistent with dynamic-image-mean-max
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

            # Multi-view summary (mean + max + cls), then project back to D
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
        """Returns (image_features, c_sem). c_sem is single-memory c_24, used after mm_proj to align with LLM last hidden states."""
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
                c_sem = c  # Single-memory c_24 for alignment with LLM hidden states after mm_proj
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
            output_attentions=True  # Enable attention map output
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
