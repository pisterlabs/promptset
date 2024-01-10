""" CLIP Model
Adapted from https://github.com/mlfoundations/open_clip
"""
import re
import json
import math
import time
import logging
from dataclasses import dataclass
from typing import Tuple, Union, Callable, Optional
from functools import partial
import numpy as np
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
import contextlib

import torch
import torch.nn.functional as F
from torch import nn

from lavis.common.registry import registry
from lavis.common.utils import get_abs_path
from lavis.models.clip_models.clip_outputs import ClipOutput, ClipOutputFeatures
from lavis.models.eva_vit import VisionTransformer
from lavis.models.base_model import BaseModel
from lavis.models.clip_models.model import openai_imagenet_template
from lavis.tasks.multimodal_classification import MultimodalClassificationTask
from lavis.models.clip_models.transform import image_transform


OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)


_MODEL_CONFIG_PATHS = [Path(__file__).parent.parent.parent / f"configs/models/eva_clip/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


cifar100_template = [
    lambda c: f"a photo of a {c}.",
]


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = (".json",)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f"*{ext}"))

    for cf in config_files:
        with open(cf, "r") as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ("embed_dim", "vision_cfg", "text_cfg")):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {
        k: v
        for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))
    }


def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())


_rescan_model_configs()  # initial populate of model config registry


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
        else:
            q = q * self.scale
            attn = torch.bmm(q, k.transpose(-1, -2))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            attn += attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(attn, v)
        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            act_layer: Callable = nn.GELU,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
    ):
        super().__init__()

        self.ln_1 = LayerNorm(d_model)
        # FIXME torchscript issues need to be resolved for custom attention
        # if scale_cosine_attn or scale_heads:
        #     self.attn = Attention(
        #        d_model, n_head,
        #        scaled_cosine=scale_cosine_attn,
        #        scale_heads=scale_heads,
        #     )
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_attn = LayerNorm(d_model) if scale_attn else nn.Identity()

        self.ln_2 = LayerNorm(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ('ln', LayerNorm(mlp_width) if scale_fc else nn.Identity()),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))

    def attention(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
        # FIXME torchscript issues need resolving for custom attention option to work
        # if self.use_torch_attn:
        #     return self.attn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]
        # else:
        #     return self.attn(x, attn_mask=attn_mask)

    def cross_attention(self, x: torch.Tensor, context: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        return self.attn(x, context, context, need_weights=False, attn_mask=attn_mask)[0]


    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.ln_attn(self.attention(self.ln_1(x), attn_mask=attn_mask))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int,  mlp_ratio: float = 4.0, act_layer: Callable = nn.GELU):
        super().__init__()
        self.width = width
        self.layers = layers

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, mlp_ratio, act_layer=act_layer)
            for _ in range(layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)
        return x


class TextTransformer(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            width: int,
            layers: int,
            heads: int,
            context_length: int,
            embed_dim: int,
            act_layer: Callable = nn.GELU,
    ):
        super().__init__()
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            act_layer=act_layer,
        )
        self.context_length = context_length
        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        self.ln_final = LayerNorm(width)

        self.text_projection = nn.Parameter(torch.empty(width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward_features(self, text: torch.Tensor):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=self.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]
        return x

    def forward(self, x: torch.Tensor):
        x = self.forward_features(x)
        if self.text_projection is not None:
            x = x @ self.text_projection
        return x

@dataclass
class CLIPVisionCfg:
    layers: Union[Tuple[int, int, int, int], int] = 12
    width: int = 768
    head_width: int = 64
    mlp_ratio: float = 4.0
    patch_size: int = 16
    image_size: Union[Tuple[int, int], int] = 224
    layer_scale_init_value: float = None
    drop_path_rate:float = 0.
    use_mean_pooling: bool = False


@dataclass
class CLIPTextCfg:
    context_length: int = 77
    vocab_size: int = 49408
    width: int = 512
    heads: int = 8
    layers: int = 12


@registry.register_model("eva_clip")
@registry.register_model("eva_clip_feature_extractor")
class EVA_CLIP(BaseModel):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "EVA-CLIP-g-336": "configs/models/eva_clip_g_336.yaml",
        "EVA-CLIP-g": "configs/models/eva_clip_g.yaml",
    }

    def __init__(
            self,
            embed_dim: int,
            vision_cfg: CLIPVisionCfg,
            text_cfg: CLIPTextCfg,
            quick_gelu: bool = False,
    ):
        from .tokenizer import tokenize

        super().__init__()

        self.tokenizer = tokenize
        self._loss = None

        if isinstance(vision_cfg, dict):
            vision_cfg = CLIPVisionCfg(**vision_cfg)
        if isinstance(text_cfg, dict):
            text_cfg = CLIPTextCfg(**text_cfg)

        # OpenAI models are pretrained w/ QuickGELU but native nn.GELU is both faster and more
        # memory efficient in recent PyTorch releases (>= 1.10).
        act_layer = QuickGELU if quick_gelu else nn.GELU

        vision_heads = vision_cfg.width // vision_cfg.head_width
        self.visual = VisionTransformer(
            img_size=vision_cfg.image_size,
            patch_size=vision_cfg.patch_size,
            num_classes=embed_dim,
            use_mean_pooling=vision_cfg.use_mean_pooling,
            init_values=vision_cfg.layer_scale_init_value,
            embed_dim=vision_cfg.width,
            depth=vision_cfg.layers,
            num_heads=vision_heads,
            mlp_ratio=vision_cfg.mlp_ratio,
            qkv_bias=True,
            drop_path_rate=vision_cfg.drop_path_rate,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )

        self.text = TextTransformer(
            vocab_size=text_cfg.vocab_size,
            width=text_cfg.width,
            layers=text_cfg.layers,
            heads=text_cfg.heads,
            context_length=text_cfg.context_length,
            embed_dim=embed_dim,
            act_layer=act_layer
        )

        self.prompt_templates = openai_imagenet_template
        self.classifier = None
        
    def maybe_autocast(self, dtype=torch.float32):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()

    def encode_image(self, image):
        return self.visual(image)

    def encode_text(self, text):
        return self.text(text)

    @property
    def loss(self):
        if self._loss is None:
            from lavis.models.clip_models.loss import ClipLoss
            from torch import distributed as dist

            self._loss = ClipLoss(
                world_size=dist.get_world_size(),
                rank=dist.get_rank(),
                local_loss=False,
                gather_with_grad=False,
                use_horovod=False,
            )

        return self._loss

    # def forward(self, image, text):
    #     if image is None:
    #         return self.encode_text(text)
    #     elif text is None:
    #         return self.encode_image(image)
    #     image_features = self.encode_image(image)
    #     image_features = F.normalize(image_features, dim=-1)

    #     text_features = self.encode_text(text)
    #     text_features = F.normalize(text_features, dim=-1)

    #     return image_features, text_features, self.text.logit_scale.exp()

    def forward(self, samples):
        image = samples.get("image")
        text = samples.get("text_input")

        if text is not None:
            text = self.tokenizer(text).to(self.device)

        if image is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_image(image)
        image_embeds = self.encode_image(image)
        image_features = F.normalize(image_embeds, dim=-1)

        text_embeds = self.encode_text(text)
        text_features = F.normalize(text_embeds, dim=-1)

        loss = self.loss(image_features, text_features, 1)

        # return image_features, text_features, self.logit_scale.exp()
        # return {"loss": loss}
        return ClipOutput(
            intermediate_output=ClipOutputFeatures(
                image_embeds=image_embeds,
                image_embeds_proj=image_features,
                text_embeds=text_embeds,
                text_embeds_proj=text_features,
            ),
            loss=loss,
            logit_scale_exp=1,
        )

    def extract_features(self, samples):
        """
        Extract features from the model for samples.

        Keys allowed are "image" and "text_input" in samples.
        If either key is missing, the corresponding features are not extracted.

        Args:
            samples: dict of samples to extract features from.

        Returns:
            ClipOutputFeatures object with features for the samples.
        """
        image = samples.get("image")
        text = samples.get("text_input")

        if text is not None:
            text = self.tokenizer(text).to(self.device)

        if image is None:
            return self.encode_text(text)
        elif text is None:
            return self.encode_image(image)

        image_embeds = self.encode_image(image)
        image_features = F.normalize(image_embeds, dim=-1)

        text_embeds = self.encode_text(text)
        text_features = F.normalize(text_embeds, dim=-1)

        return ClipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
        )

    def predict(self, samples):
        image = samples["image"]
        targets = samples["label"]

        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)

        logits = 100.0 * image_features @ self.classifier

        return {"predictions": logits, "targets": targets}

    def before_evaluation(self, dataset, task_type, **kwargs):
        if task_type == MultimodalClassificationTask:
            self.classifier = self.zero_shot_classifier(
                classnames=dataset.classnames,
                templates=self.prompt_templates,
            )

    def zero_shot_classifier(self, classnames, templates):
        with torch.no_grad():
            zeroshot_weights = []
            for classname in classnames:
                texts = [
                    template(classname) for template in templates
                ]  # format with class
                texts = self.tokenizer(texts).to(self.device)  # tokenize

                class_embeddings = self.encode_text(texts)
                class_embedding = F.normalize(class_embeddings, dim=-1).mean(dim=0)
                class_embedding /= class_embedding.norm()
                zeroshot_weights.append(class_embedding)
            zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(self.device)
        return zeroshot_weights

    @classmethod
    def default_config_path(cls, model_type="base"):
        model_type = "EVA-CLIP-g" if model_type == "base" else model_type

        assert (
            model_type in cls.PRETRAINED_MODEL_CONFIG_DICT
        ), "Unknown model type {}. \n Available types: {}".format(
            model_type, cls.PRETRAINED_MODEL_CONFIG_DICT.keys()
        )
        return get_abs_path(cls.PRETRAINED_MODEL_CONFIG_DICT[model_type])

    @classmethod
    def from_config(cls, cfg=None):
        model_name = cfg.model_type
        pretrained = cfg.pretrained

        precision = cfg.get("precision", "fp32")

        return create_model(
            model_name=model_name, pretrained=pretrained, precision=precision
        )

    def zero_shot_predict(self, image_path, categories):
        assert isinstance(
            categories, list
        ), f"categories must be a list, got {type(categories)}."
        assert os.path.exists(image_path), f"File {image_path} does not exist."

        from lavis.processors.clip_processors import ClipImageEvalProcessor
        from PIL import Image

        image_preprocess = ClipImageEvalProcessor()
        image = image_preprocess(Image.open(image_path)).unsqueeze(0)

        text = self.tokenizer(categories)

        with torch.no_grad():
            image_features = self.encode_image(image)
            text_features = self.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            print("Label probs:", text_probs)  # prints: [[1., 0., 0.]]

    def compute_sim_matrix(self, data_loader, **kwargs):
        logging.info("Computing features for evaluation...")
        start_time = time.time()

        texts = data_loader.dataset.text
        num_text = len(texts)
        text_bs = 256
        text_features = []

        for i in range(0, num_text, text_bs):

            text = texts[i : min(num_text, i + text_bs)]
            text_input = self.tokenizer(text).to(self.device)

            text_feat = self.encode_text(text_input)
            text_feat = F.normalize(text_feat, dim=-1)

            text_features.append(text_feat)

        text_features = torch.cat(text_features, dim=0)

        image_features = []
        for samples in data_loader:
            image = samples["image"]

            image = image.to(self.device)
            image_feat = self.encode_image(image)
            image_feat = F.normalize(image_feat, dim=-1)

            image_features.append(image_feat)

        image_features = torch.cat(image_features, dim=0)

        sims_matrix_i2t = image_features @ text_features.t()
        sims_matrix_t2i = sims_matrix_i2t.t()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Evaluation time {}".format(total_time_str))

        return sims_matrix_i2t.cpu().numpy(), sims_matrix_t2i.cpu().numpy()


def load_state_dict(checkpoint_path: str, map_location: str='cpu', model_key='model|module|state_dict'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)

    for mk in model_key.split('|'):
        if isinstance(checkpoint, dict) and mk in checkpoint:
            state_dict = checkpoint[mk]
            break
        else:
            state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def interpolate_pos_embed(model, checkpoint_model):

    # for k, v in checkpoint_model.items():
    #     print(k, v.shape)
    # exit()
    if 'visual.pos_embed' in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model['visual.pos_embed'].float()
        embedding_size = pos_embed_checkpoint.shape[-1]
        num_patches = model.visual.patch_embed.num_patches
        num_extra_tokens = model.visual.pos_embed.shape[-2] - num_patches
        # height (== width) for the checkpoint position embedding
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
        # height (== width) for the new position embedding
        new_size = int(num_patches ** 0.5)
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Position interpolate from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['visual.pos_embed'] = new_pos_embed


def load_checkpoint(model, checkpoint_path, model_key="model|module|state_dict", strict=True):
    state_dict = load_state_dict(checkpoint_path, model_key=model_key)

    interpolate_pos_embed(model, state_dict)

    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    print(incompatible_keys)
    return incompatible_keys


def create_model(
        model_name: str,
        pretrained: str = '',
        precision: str = 'fp32',
        device: torch.device = torch.device('cpu'),
        force_quick_gelu: bool = False,
):
    model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names

    if model_name in _MODEL_CONFIGS:
        logging.info(f'Loading {model_name} model config.')
        model_cfg = deepcopy(_MODEL_CONFIGS[model_name])
    else:
        logging.error(f'Model config for {model_name} not found; available models {list_models()}.')
        raise RuntimeError(f'Model config for {model_name} not found.')

    if force_quick_gelu:
        # override for use of QuickGELU on non-OpenAI transformer models
        model_cfg["quick_gelu"] = True

    model = EVA_CLIP(**model_cfg)

    load_checkpoint(model, pretrained)
                
    model.to(device=device)
    if precision == "fp16":
        assert device.type != 'cpu'
        convert_weights_to_fp16(model)

    # set image / mean metadata from pretrained_cfg if available, or use default
    model.visual.image_mean = OPENAI_DATASET_MEAN
    model.visual.image_std = OPENAI_DATASET_STD

    return model


def create_model_and_transforms(
    model_name: str,
    pretrained: str = "",
    precision: str = "fp32",
    device: torch.device = torch.device("cpu"),
    jit: bool = False,
    force_quick_gelu: bool = False,
    pretrained_image: bool = False,
):
    model = create_model(
        model_name,
        pretrained,
        precision,
        device,
        force_quick_gelu=force_quick_gelu,
    )
    preprocess_train = image_transform(model.visual.image_size, is_train=True)
    preprocess_val = image_transform(model.visual.image_size, is_train=False)
    return model, preprocess_train, preprocess_val


def convert_weights_to_fp16(model: nn.Module):
    """Convert applicable model parameters to fp16"""

    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, (nn.MultiheadAttention, Attention)):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj"]:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)