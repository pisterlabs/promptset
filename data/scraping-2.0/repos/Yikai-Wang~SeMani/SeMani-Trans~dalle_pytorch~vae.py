import io
import sys
import os, sys
import requests
import PIL
import warnings
import os
import hashlib
import urllib 
import yaml
from pathlib import Path
from tqdm import tqdm
from math import sqrt, log
from omegaconf import OmegaConf 
from taming.models.vqgan import VQModel
# from taming.models.vqgan import GumbelVQ

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange

# from dalle_pytorch import distributed_utils

# constants

CACHE_PATH = os.path.expanduser("~/.cache/dalle")

OPENAI_VAE_ENCODER_PATH = 'https://cdn.openai.com/dall-e/encoder.pkl'
OPENAI_VAE_DECODER_PATH = 'https://cdn.openai.com/dall-e/decoder.pkl'

VQGAN_VAE_PATH = 'https://heibox.uni-heidelberg.de/f/140747ba53464f49b476/?dl=1'
VQGAN_VAE_CONFIG_PATH = 'https://heibox.uni-heidelberg.de/f/6ecf2af6c658432c8298/?dl=1'

# helpers methods

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def load_model(path):
    with open(path, 'rb') as f:
        return torch.load(f, map_location = torch.device('cpu'))

def map_pixels(x, eps = 0.1):
    return (1 - 2 * eps) * x + eps

def unmap_pixels(x, eps = 0.1):
    return torch.clamp((x - eps) / (1 - 2 * eps), 0, 1)


# TODO. Copy from the latest version of VQGAN github

# pretrained Discrete VAE from OpenAI

class OpenAIDiscreteVAE(nn.Module):
    def __init__(self, filepath = None):
        super().__init__()

        if filepath is not None:
            self.enc = load_model(os.path.join(filepath, 'encoder.pkl'))
            self.dec = load_model(os.path.join(filepath, 'decoder.pkl'))
        else:
            # self.enc = load_model(download(OPENAI_VAE_ENCODER_PATH))
            # self.dec = load_model(download(OPENAI_VAE_DECODER_PATH))
            raise NotImplemented

        self.num_layers = 3
        self.image_size = 256
        self.num_tokens = 8192

    @torch.no_grad()
    def get_codebook_indices(self, img):
        img = map_pixels(img)
        z_logits = self.enc.blocks(img)
        z = torch.argmax(z_logits, dim = 1)
        return rearrange(z, 'b h w -> b (h w)')

    def decode(self, img_seq):
        if img_seq.dim() == 2:
            b, n = img_seq.shape
            img_seq = rearrange(img_seq, 'b (h w) -> b h w', h = int(sqrt(n)))

            z = F.one_hot(img_seq, num_classes = self.num_tokens)
            z = rearrange(z, 'b h w c -> b c h w').float()
        elif img_seq.dim() == 4:
            z = img_seq
        else:
            raise NotImplemented
        x_stats = self.dec(z).float()
        x_rec = unmap_pixels(torch.sigmoid(x_stats[:, :3]))
        return x_rec

    def forward(self, img):
        raise NotImplemented

# VQGAN from Taming Transformers paper
# https://arxiv.org/abs/2012.09841

class VQGanVAE(nn.Module):
    def __init__(self, vqgan_model_path, vqgan_config_path):
        super().__init__()

        if vqgan_model_path is None:
            model_filename = 'vqgan.1024.model.ckpt'
            config_filename = 'vqgan.1024.config.yml'
            # download(VQGAN_VAE_CONFIG_PATH, config_filename)
            # download(VQGAN_VAE_PATH, model_filename)
            raise NotImplemented
            config_path = str(Path(CACHE_PATH) / config_filename)
            model_path = str(Path(CACHE_PATH) / model_filename)
        else:
            model_path = vqgan_model_path
            config_path = vqgan_config_path

        config = OmegaConf.load(config_path)
        if vqgan_model_path.find('gumbel') != -1:
            model = GumbelVQ(**config.model.params)
            self.gumbel = True
        else:
            model = VQModel(**config.model.params)
            self.gumbel = False

        state = torch.load(model_path, map_location = 'cpu')['state_dict']
        model.load_state_dict(state, strict = False)

        print(f"Loaded VQGAN from {model_path} and {config_path}")

        self.model = model

        self.num_layers = int(log(config.model.params.ddconfig.attn_resolutions[0])/log(2))
        self.image_size = 256
        self.num_tokens = config.model.params.n_embed

    @torch.no_grad()
    def fetch_probability(self, img):
        b = img.shape[0]
        img = (2 * img) - 1
        # self.model.encode
        h = self.model.encoder(img)
        h = self.model.quant_conv(h)
        # quantize
        z = h
        z = z.permute(0, 2, 3, 1).contiguous()
        z_flattened = z.view(-1, self.model.quantize.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(self.model.quantize.embedding.weight ** 2, dim=1) - 2 * \
            torch.matmul(z_flattened, self.model.quantize.embedding.weight.t())  # batch*height*width, -1
        d = d.view(b, -1, d.size(-1))
        vs, idx = torch.topk(d, k=d.size(-1), dim=-1)  # (bsz, seq, -1)
        return idx  # rearrange(indices, '(b n) () -> b n', b = b)

    @torch.no_grad()
    def get_codebook_indices(self, img):
        b = img.shape[0]
        img = (2 * img) - 1
        _, _, [_, _, indices] = self.model.encode(img)  # (B, C, H, W) -> (B, H, W, C) -> (B*H*W, C)
        if indices.dim() == 3:
            indices = indices.view(-1, 1)
        elif indices.dim() == 1:
            indices = indices.unsqueeze(1)
        return rearrange(indices, '(b n) () -> b n', b = b)

    def decode(self, img_seq):
        raise NotImplemented
        # if img_seq.dim() == 2:
        #     b, n = img_seq.shape
        #     one_hot_indices = F.one_hot(img_seq, num_classes = self.num_tokens).to(self.model.quantize.embedding.weight)
        #     z = torch.matmul(one_hot_indices, self.model.quantize.embedding.weight)
        #     z = rearrange(z, 'b (h w) c -> b c h w', h = int(sqrt(n)))
        # elif img_seq.dim() == 4:
        #     soft_one_hot_indices = img_seq  # (b, c, h, w)
        #     soft_one_hot_indices = rearrange(soft_one_hot_indices, 'b c h w -> b h w c')
        #
        #     z = torch.matmul(soft_one_hot_indices, self.model.quantize.embedding.weight)
        #     z = rearrange(z, 'b h w c -> b c h w')
        # else:
        #     raise NotImplementedError
        #
        # img = self.model.decode(z)
        # img = (img.clamp(-1., 1.) + 1) * 0.5
        # return img

    def forward(self, img_seq):
        if img_seq.dim() == 2:
            b, n = img_seq.shape
            if self.gumbel:
                one_hot_indices = F.one_hot(img_seq, num_classes = self.num_tokens).to(self.model.quantize.embed.weight)
                z = torch.matmul(one_hot_indices, self.model.quantize.embed.weight)
            else:
                one_hot_indices = F.one_hot(img_seq, num_classes=self.num_tokens).to(
                    self.model.quantize.embedding.weight)
                z = torch.matmul(one_hot_indices, self.model.quantize.embedding.weight)
            z = rearrange(z, 'b (h w) c -> b c h w', h = int(sqrt(n)))
        elif img_seq.dim() == 4:
            soft_one_hot_indices = img_seq  # (b, c, h, w)
            soft_one_hot_indices = rearrange(soft_one_hot_indices, 'b c h w -> b h w c')
            if self.gumbel:
                z = torch.matmul(soft_one_hot_indices, self.model.quantize.embed.weight)
            else:
                z = torch.matmul(soft_one_hot_indices, self.model.quantize.embedding.weight)
            z = rearrange(z, 'b h w c -> b c h w')
        else:
            raise NotImplementedError

        img = self.model.decode(z)
        img = (img.clamp(-1., 1.) + 1) * 0.5

        return img
