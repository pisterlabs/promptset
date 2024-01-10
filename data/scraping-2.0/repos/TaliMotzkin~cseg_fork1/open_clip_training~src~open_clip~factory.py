import json
import logging
import os
import pathlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Optional, Tuple, List

import torch
from torch import nn
from torch.nn import functional as F

from .model import CLIP, convert_weights_to_fp16, resize_pos_embed
from .openai import load_openai_model
from .pretrained import get_pretrained_url, download_pretrained
from .transform import image_transform
from .transformer_adapter import TwoWayTransformer
from .mask_decoder import MaskDecoder
import numpy as np

_MODEL_CONFIG_PATHS = [Path(__file__).parent / f"model_configs/"]
_MODEL_CONFIGS = {}  # directory (model_name: config) of model architecture configs


def _natural_key(string_):
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def _rescan_model_configs():
    global _MODEL_CONFIGS

    config_ext = ('.json',)
    config_files = []
    for config_path in _MODEL_CONFIG_PATHS:
        if config_path.is_file() and config_path.suffix in config_ext:
            config_files.append(config_path)
        elif config_path.is_dir():
            for ext in config_ext:
                config_files.extend(config_path.glob(f'*{ext}'))

    for cf in config_files:
        with open(cf, 'r') as f:
            model_cfg = json.load(f)
            if all(a in model_cfg for a in ('embed_dim', 'vision_cfg', 'text_cfg')):
                _MODEL_CONFIGS[cf.stem] = model_cfg

    _MODEL_CONFIGS = {k: v for k, v in sorted(_MODEL_CONFIGS.items(), key=lambda x: _natural_key(x[0]))}


_rescan_model_configs()  # initial populate of model config registry


def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model, checkpoint_path, strict=True):
    state_dict = load_state_dict(checkpoint_path)
    resize_pos_embed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def create_model(
        model_name: str,
        pretrained: str = '',
        precision: str = 'fp32',
        device: torch.device = torch.device('cpu'),
        jit: bool = False,
        force_quick_gelu: bool = False,
        pretrained_image: bool = False,
        mask_emb_depth: int = 0,
):
    model_name = model_name.replace('/', '-')  # for callers using old naming with / in ViT names

    if pretrained.lower() == 'openai':
        logging.info(f'Loading pretrained {model_name} from OpenAI.')
        model = load_openai_model(model_name, device=device, jit=jit)
        # See https://discuss.pytorch.org/t/valueerror-attemting-to-unscale-fp16-gradients/81372
        if precision == "amp" or precision == "fp32":
            model = model.float()
    else:
        if model_name in _MODEL_CONFIGS:
            logging.info(f'Loading {model_name} model config.')
            model_cfg = deepcopy(_MODEL_CONFIGS[model_name])
        else:
            logging.error(f'Model config for {model_name} not found; available models {list_models()}.')
            raise RuntimeError(f'Model config for {model_name} not found.')

        if force_quick_gelu:
            # override for use of QuickGELU on non-OpenAI transformer models
            model_cfg["quick_gelu"] = True

        if mask_emb_depth > 0:
            model_cfg['vision_cfg']['mask_emb_depth'] = mask_emb_depth

        if pretrained_image:
            if 'timm_model_name' in model_cfg.get('vision_cfg', {}):
                # pretrained weight loading for timm models set via vision_cfg
                model_cfg['vision_cfg']['timm_model_pretrained'] = True
            else:
                assert False, 'pretrained image towers currently only supported for timm models'

        model = CLIP(**model_cfg)
        
        if pretrained:
            checkpoint_path = ''
            url = get_pretrained_url(model_name, pretrained)
            if url:
                checkpoint_path = download_pretrained(url)
            elif os.path.exists(pretrained):
                checkpoint_path = pretrained

            if checkpoint_path:
                logging.info(f'Loading pretrained {model_name} weights ({pretrained}).')
                try:
                    load_checkpoint(model, checkpoint_path)
                except:
                    load_checkpoint(model, checkpoint_path, strict=False)
                    logging.info("The keys in the checkpoint_path don't match that of model, make sure that you are doing mask prompt tuning!")
            else:
                logging.warning(f'Pretrained weights ({pretrained}) not found for model {model_name}.')
                raise RuntimeError(f'Pretrained weights ({pretrained}) not found for model {model_name}.')

        model.to(device=device)
        if precision == "fp16":
            assert device.type != 'cpu'
            convert_weights_to_fp16(model)

        if jit:
            model = torch.jit.script(model)

    return model

def create_cseg_model(
        model_name: str,
        pretrained: str = '',
        precision: str = 'fp32',
        device: torch.device = torch.device('cpu'),
        jit: bool = False,
        force_quick_gelu: bool = False,
        pretrained_image: bool = False,
        mask_emb_depth: int = 0,
):
    mask_model = create_model(
        model_name, pretrained, precision, device, jit,
        force_quick_gelu=force_quick_gelu,
        pretrained_image=pretrained_image,
        mask_emb_depth=mask_emb_depth)
    image_model = create_model(
        model_name, pretrained, precision, device, jit,
        force_quick_gelu=force_quick_gelu,
        pretrained_image=pretrained_image,
        mask_emb_depth=0)
    model = ClipEnsembler(image_model, mask_model)
    return model


def create_model_and_transforms(
        model_name: str,
        pretrained: str = '',
        precision: str = 'fp32',
        device: torch.device = torch.device('cpu'),
        jit: bool = False,
        force_quick_gelu: bool = False,
        pretrained_image: bool = False,
        mean: Optional[Tuple[float, ...]] = None,
        std: Optional[Tuple[float, ...]] = None,
        scale: Optional[Tuple[float, ...]] = None,
        erosion: bool = False,
        with_mask: bool = False,
        mask_emb_depth: int = 0,
):
    model = create_cseg_model(
        model_name, pretrained, precision, device, jit,
        force_quick_gelu=force_quick_gelu,
        pretrained_image=pretrained_image,
        mask_emb_depth=mask_emb_depth)
    model = model.to(device=device)
    preprocess_train = image_transform(model.clip_model.visual.image_size, is_train=True, mean=mean, std=std, 
                                       scale=scale, erosion=erosion, with_mask=with_mask)
    preprocess_val = image_transform(model.clip_model.visual.image_size, is_train=False, mean=mean, std=std, with_mask=with_mask)
    return model, preprocess_train, preprocess_val


def list_models():
    """ enumerate available model architectures based on config files """
    return list(_MODEL_CONFIGS.keys())


def add_model_config(path):
    """ add model config path or file and update registry """
    if not isinstance(path, Path):
        path = Path(path)
    _MODEL_CONFIG_PATHS.append(path)
    _rescan_model_configs()


class ClipEnsembler(nn.Module):
    def __init__(
        self,
        clip_model,
        clip_mask_model, 
    ):
        super().__init__()
        self.clip_model_reg = clip_model
        self.clip_model = clip_mask_model
        vit_embed_dim = 1024
        self.image_embedding_size = vit_embed_dim
        self.non_object_embedding = nn.Parameter(
            torch.empty(1, self.clip_model.text_projection.shape[-1])
        )
        nn.init.normal_(
            self.non_object_embedding.data,
            std=self.clip_model.transformer.width ** -0.5,
        )

        self.mask_decoder=MaskDecoder(
            num_multimask_outputs=1, # Find a way to make this dynamic
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=vit_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=vit_embed_dim,
        )
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
        text
    ):
        masked_vit_features = None
        regions = mask
        #print('IMAGE input size: ', image.shape) # Num Images x 256 x grid x grid
        #print('MASK input size: ', regions.shape) # Num Masks Per Image x 256 x grid x grid
        if isinstance(regions, list):
            assert NotImplementedError
            masked_features = torch.cat(
                [self.get_image_features(image_i) for image_i in regions], dim=0
            )
        else:
            masked_features, masked_vit_features, cls_embed = self.get_mask_embed(regions)

        image_features, image_vit_features, positional_encoding = self.get_image_embed(image)
        #print('IMAGE ViT FEATURES: ', image_vit_features.shape) # Num Images x 256 x grid x grid
        #print('MASK ViT FEATURES: ', masked_vit_features.shape) # Num Masks Per Image x 256 x grid x grid
        b, c, grid, _ = image_vit_features.shape
        n, _, _, _ = masked_vit_features.shape
        b_enc, c_enc, grid_enc, _ = positional_encoding.shape
        updated_mask_vit_features = self.mask_decoder(
                image_embeddings=image_vit_features.reshape(b, c, grid**2),
                image_pe=positional_encoding.reshape(b_enc, c_enc, grid_enc**2),
                mask_embeddings=masked_vit_features.reshape(n, c, grid**2),
                cls_embed=cls_embed.reshape(n, 1, grid**2),
            )
        updated_mask_features = self.get_mask_features(updated_mask_vit_features)
        text_features = self.clip_model.encode_text(text)
        return updated_mask_features, text_features, self.logit_scale.exp()
    

    def encode_image(
        self,
        image: torch.Tensor,
        mask: torch.Tensor,
    ):  
        masked_vit_features = None
        regions = mask
        #print('IMAGE input size: ', image.shape) # Num Images x 256 x grid x grid
        #print('MASK input size: ', regions.shape) # Num Masks Per Image x 256 x grid x grid
        if isinstance(regions, list):
            assert NotImplementedError
            masked_features = torch.cat(
                [self.get_image_features(image_i) for image_i in regions], dim=0
            )
        else:
            masked_features, masked_vit_features, cls_embed = self.get_mask_embed(regions)

        image_features, image_vit_features, positional_encoding = self.get_image_embed(image)
        #print('IMAGE ViT FEATURES: ', image_vit_features.shape) # Num Images x 256 x grid x grid
        #print('MASK ViT FEATURES: ', masked_vit_features.shape) # Num Masks Per Image x 256 x grid x grid
        b, c, grid, _ = image_vit_features.shape
        n, _, _, _ = masked_vit_features.shape
        b_enc, c_enc, grid_enc, _ = positional_encoding.shape
        updated_mask_vit_features = self.mask_decoder(
                image_embeddings=image_vit_features.reshape(b, c, grid**2),
                image_pe=positional_encoding.reshape(b_enc, c_enc, grid_enc**2),
                mask_embeddings=masked_vit_features.reshape(n, c, grid**2),
                cls_embed=cls_embed.reshape(n, 1, grid**2),
            )

        updated_mask_features = self.get_mask_features(updated_mask_vit_features)
        return updated_mask_features
    # -----------------------------------------------------------------------------
    def get_mask_embed(self, image, region_masks=None):
      image_features, vit_embed, _  = self.clip_model.visual.get_vit_embedding(image, region_masks)
      cls_embed = vit_embed[:,:1]
      #print('CLS EMBED: ', cls_embed.shape)
      #cls_embed = cls_embed.reshape(cls_embed.size(0), cls_embed.size(1), cls_embed.size(2)**2).permute(0,2,1)
      vit_embed = vit_embed[:,1:]
      return image_features, vit_embed, cls_embed

    # -----------------------------------------------------------------------------
    def get_mask_features(self, vit_region_embed: torch.Tensor):
      image_features = self.clip_model.visual.get_clip_embedding(vit_region_embed)
      image_features = image_features / image_features.norm(dim=-1, keepdim=True)
      return image_features

    # -----------------------------------------------------------------------------
    def get_image_embed(self, image: torch.Tensor):
        image_features, vit_embed, positional_encoding  = self.clip_model_reg.visual.get_vit_embedding(image)
        vit_embed = vit_embed[:,1:]
        return image_features, vit_embed, positional_encoding

    # -----------------------------------------------------------------------------
    def get_image_features(self, vit_embed: torch.Tensor):
        image_features  = self.clip_model_reg.visual.get_clip_embedding(vit_embed)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        return image_features

    def normalize_feature(self, feat: torch.Tensor):
        return feat / feat.norm(dim=-1, keepdim=True)
