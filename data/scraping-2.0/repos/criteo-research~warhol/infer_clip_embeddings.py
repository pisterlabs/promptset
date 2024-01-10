import argparse
from pathlib import Path
import time
from glob import glob
import os
import shutil
import numpy as np
from tqdm import tqdm

import torch
import wandb  # Quit early if user doesn't have wandb installed.
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F

from dalle_pytorch import OpenAIDiscreteVAE, VQGanVAE, DiscreteVAE, WARHOL
from dalle_pytorch import distributed_utils
from dalle_pytorch.loader import TextImageDataset
from dalle_pytorch.tokenizer import tokenizer, HugTokenizer, ChineseTokenizer, YttmTokenizer
import clip

# libraries needed for webdataset support
import webdataset as wds
from torchvision import transforms as T
from PIL import Image
from io import BytesIO


# argument parsing

parser = argparse.ArgumentParser()

group = parser.add_mutually_exclusive_group(required=False)

group.add_argument('--vae_path', type=str,
                   help='path to your trained discrete VAE')

group.add_argument('--from_smaller_model', type=str,
                   help='path to your trained DALL-E')

group.add_argument('--warhol_path', type=str,
                   help='path to your partially trained WARHOL')

parser.add_argument('--ft_next_prod_module', dest='ft_next_prod_module', action='store_true')

parser.add_argument('--inferring_clip_embeddings', dest='inferring_clip_embeddings', action='store_true')

parser.add_argument('--use_of_clip_embed', type = str, default='', 
    help = 'Define what CLIP embeddings do we use for conditioning: img, txt or both')

parser.add_argument('--vqgan_model_path', type=str, default = None,
                   help='path to your trained VQGAN weights. This should be a .ckpt file. (only valid when taming option is enabled)')

parser.add_argument('--vqgan_config_path', type=str, default = None,
                   help='path to your trained VQGAN config. This should be a .yaml file. (only valid when taming option is enabled)')

parser.add_argument('--image_text_folder', type=str, required=True,
                    help='path to your folder of images and text for learning the DALL-E')

parser.add_argument('--negative_samples_path', type=str, default=None,
                    help='path to your folder of clip embeddings of negative samples')

parser.add_argument('--wds', type = str, default='', 
    help = 'Comma separated list of WebDataset (1) image and (2) text column names. Must contain 2 values, e.g. img,cap.'
)

parser.add_argument('--truncate_captions', dest='truncate_captions', action='store_true',
                    help='Captions passed in which exceed the max token length will be truncated if this is set.')

parser.add_argument('--random_resize_crop_lower_ratio', dest='resize_ratio', type=float, default=0.75,
                    help='Random resized crop lower ratio')

parser.add_argument('--chinese', dest='chinese', action='store_true')

parser.add_argument('--taming', dest='taming', action='store_true')

parser.add_argument('--hug', dest='hug', action='store_true')

parser.add_argument('--bpe_path', type=str,
                    help='path to your BPE json file')

parser.add_argument('--warhol_output_file_name', type=str, default = "warhol",
                    help='output_file_name')

parser.add_argument('--fp16', action='store_true',
                    help='(experimental) - Enable DeepSpeed 16 bit precision. Reduces VRAM.')


parser.add_argument('--amp', action='store_true',
	help='Apex "O1" automatic mixed precision. More stable than 16 bit precision. Can\'t be used in conjunction with deepspeed zero stages 1-3.')

parser.add_argument('--wandb_name', default='warhol_train_transformer',
                    help='Name W&B will use when saving results.\ne.g. `--wandb_name "coco2017-full-sparse"`')

parser.add_argument('--wandb_entity', default=None,
                    help='(optional) Name of W&B team/entity to log to.')

parser.add_argument('--stable_softmax', dest='stable_softmax', action='store_true',
                    help='Prevent values from becoming too large during softmax. Helps with stability in fp16 and Mixture of Quantization training.')

parser = distributed_utils.wrap_arg_parser(parser)

train_group = parser.add_argument_group('Training settings')

train_group.add_argument('--flops_profiler', dest = 'flops_profiler', action='store_true', help = 'Exits after printing detailed flops/runtime analysis of forward/backward')

train_group.add_argument('--epochs', default = 20, type = int, help = 'Number of epochs')

train_group.add_argument('--save_every_n_steps', default = 1000, type = int, help = 'Save a checkpoint every n steps')

train_group.add_argument('--keep_n_checkpoints', default = None, type = int, help = '(Careful) Deletes old deepspeed checkpoints if there are more than n')

train_group.add_argument('--batch_size', default = 1, type = int, help = 'Batch size')

train_group.add_argument('--ga_steps', default = 1, type = int, help = 'Number of steps to accumulate gradients across per each iteration. DeepSpeed only.')

train_group.add_argument('--learning_rate', default = 3e-4, type = float, help = 'Learning rate')

train_group.add_argument('--clip_grad_norm', default = 0.5, type = float, help = 'Clip gradient norm')

train_group.add_argument('--lr_decay', dest = 'lr_decay', action = 'store_true')

model_group = parser.add_argument_group('Model settings')

model_group.add_argument('--use_next_prod_module', dest = 'use_next_prod_module', action='store_true')

model_group.add_argument('--dim', default = 512, type = int, help = 'Model dimension')

model_group.add_argument('--text_seq_len', default = 256, type = int, help = 'Text sequence length')

model_group.add_argument('--depth', default = 2, type = int, help = 'Model depth')

model_group.add_argument('--heads', default = 8, type = int, help = 'Model number of heads')

model_group.add_argument('--dim_head', default = 64, type = int, help = 'Model head dimension')

train_group.add_argument('--ff_dropout', default = 0.0, type = float, help = 'Feed forward dropout.')

train_group.add_argument('--attn_dropout', default = 0.0, type = float, help = 'Feed forward dropout.')

model_group.add_argument('--reversible', dest = 'reversible', action='store_true')

model_group.add_argument('--loss_img_weight', default = 7, type = int, help = 'Image loss weight')

model_group.add_argument('--attn_types', default = 'full', type = str, help = 'comma separated list of attention types. attention type can be: full or sparse or axial_row or axial_col or conv_like.')

args = parser.parse_args()

# helpers

def exists(val):
    return val is not None

def get_trainable_params(model):
    return [params for params in model.parameters() if params.requires_grad]

def cp_path_to_dir(cp_path, tag):
    """Convert a checkpoint path to a directory with `tag` inserted.
    If `cp_path` is already a directory, return it unchanged.
    """
    if not isinstance(cp_path, Path):
        cp_path = Path(cp_path)
    if cp_path.is_dir():
        return cp_path
    path_sans_extension = cp_path.parent / cp_path.stem
    cp_dir = Path(f'{path_sans_extension}-{tag}-cp')
    return cp_dir

# constants
WEBDATASET_IMAGE_TEXT_COLUMNS = tuple(args.wds.split(','))
ENABLE_WEBDATASET = True if len(WEBDATASET_IMAGE_TEXT_COLUMNS) == 2 else False
USE_NEG_SAMPLES = args.negative_samples_path is not None
WARHOL_OUTPUT_FILE_NAME = args.warhol_output_file_name + ".pt"

VAE_PATH = args.vae_path
VQGAN_MODEL_PATH = args.vqgan_model_path
VQGAN_CONFIG_PATH = args.vqgan_config_path
WARHOL_PATH = args.warhol_path
RESUME = exists(WARHOL_PATH)
SMALLER_MODEL_PATH = args.from_smaller_model
FROM_SMALLER_MODEL = exists(SMALLER_MODEL_PATH)

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

LEARNING_RATE = args.learning_rate
GRAD_CLIP_NORM = args.clip_grad_norm
LR_DECAY = args.lr_decay
SAVE_EVERY_N_STEPS = args.save_every_n_steps
KEEP_N_CHECKPOINTS = args.keep_n_checkpoints

USE_NEXT_PROD_MODULE = args.use_next_prod_module
MODEL_DIM = args.dim
TEXT_SEQ_LEN = args.text_seq_len
DEPTH = args.depth
HEADS = args.heads
DIM_HEAD = args.dim_head
REVERSIBLE = args.reversible
LOSS_IMG_WEIGHT = args.loss_img_weight
FF_DROPOUT = args.ff_dropout
ATTN_DROPOUT = args.attn_dropout
STABLE = args.stable_softmax
ATTN_TYPES = tuple(args.attn_types.split(','))
DEEPSPEED_CP_AUX_FILENAME = 'auxiliary.pt'

def preprocess(image_input):
    image_mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])
    image_std = torch.tensor([0.26862954, 0.26130258, 0.27577711])
    image_input = F.interpolate(image_input, size=224)
    image_input -= image_mean[:, None, None]
    image_input /= image_std[:, None, None]
    return image_input

# quit early if you used the wrong folder name
assert Path(args.image_text_folder).exists(), f'The path {args.image_text_folder} was not found.'
if args.negative_samples_path is not None:
    assert Path(args.negative_samples_path).exists(), f'The path {args.negative_samples_path} was not found.'

IMAGE_SIZE=256
is_shuffle = False
imagepreproc = T.Compose([
    T.Lambda(lambda img: img.convert('RGB')
    if img.mode != 'RGB' else img),
    T.RandomResizedCrop(IMAGE_SIZE,
                        scale=(args.resize_ratio, 1.),
                        ratio=(1., 1.)),
    T.ToTensor(),
])

def imagetransform(b):
    return Image.open(BytesIO(b))

def tokenize(s):
    return tokenizer.tokenize(
        s.decode('utf-8'),
        TEXT_SEQ_LEN,
        truncate_text=args.truncate_captions).squeeze(0)

ds = TextImageDataset(
    args.image_text_folder,
    text_len=TEXT_SEQ_LEN,
    image_size=IMAGE_SIZE,
    resize_ratio=args.resize_ratio,
    truncate_captions=args.truncate_captions,
    tokenizer=tokenizer,
    shuffle=is_shuffle,
    negatives_path=args.negative_samples_path,
    clip_embeddings=not args.inferring_clip_embeddings
)
assert len(ds) > 0, 'dataset is empty'

#data_sampler = torch.utils.data.distributed.DistributedSampler(ds)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=is_shuffle, drop_last=True, sampler=None)
device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
for epoch in range(1):
    for i, batch in tqdm(enumerate(dl)):
        key = ds.keys[i]
        if args.inferring_clip_embeddings:
            (text, images) = batch
            with torch.no_grad():
                emb_im = clip_model.encode_image(preprocess(images).to(device))
                emb_im /= emb_im.norm(dim=-1, keepdim=True)
                emb_txt = clip_model.encode_text(text[:, :77].to(device))
                emb_txt /= emb_txt.norm(dim=-1, keepdim=True)
                emb_im = emb_im.detach().cpu().numpy().flatten()
                emb_txt = emb_txt.detach().cpu().numpy().flatten()
                emb = np.concatenate((emb_im, emb_txt))
                filename = args.image_text_folder + str(key)
                np.save(filename, emb)
                