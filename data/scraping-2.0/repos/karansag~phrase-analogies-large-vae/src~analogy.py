from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import glob
import logging
import os
import pickle
import random
import sys

import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import (
    DataLoader,
    Dataset,
    SequentialSampler,
    RandomSampler,
    TensorDataset,
)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

# TODO fix this to not manually append PYTHONPATH?
OPTIMUS_ROOT_PATH = os.path.abspath("../Optimus/code")
if OPTIMUS_ROOT_PATH not in sys.path:
    sys.path.append(OPTIMUS_ROOT_PATH)

OPTIMUS_EXAMPLES_PATH = os.path.abspath("../Optimus/code/examples/big_ae")
if OPTIMUS_EXAMPLES_PATH not in sys.path:
    sys.path.append(OPTIMUS_EXAMPLES_PATH)

from pytorch_transformers import (
    GPT2Config,
    OpenAIGPTConfig,
    XLNetConfig,
    TransfoXLConfig,
    BertConfig,
)
from pytorch_transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2ForLatentConnector
from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from pytorch_transformers import XLNetLMHeadModel, XLNetTokenizer
from pytorch_transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from pytorch_transformers import BertForLatentConnector, BertTokenizer

from collections import defaultdict, namedtuple
from modules import VAE
from utils import TextDataset_Split, TextDataset_2Tokenizers, BucketingDataLoader
import run_latent_generation as runl
from util import get_device

LATENT_SIZE_LARGE = 768
MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2ForLatentConnector, GPT2Tokenizer),
    "bert": (BertConfig, BertForLatentConnector, BertTokenizer),
}

CODER_TYPE_TO_NAME = {"gpt2": "gpt2", "bert": "bert-base-cased"}

# Set this when running experiments
# TODO: make a parameter?
# OUTPUT_DIR = os.path.abspath("../../data/snli-b1/checkpoint-31250/")
# Set the output dir here. E.g.,
# export OUTPUT_DIR="/scratch/MYUSER/project-repo/pretrained_models/snli-b1/checkpoint-31250/"
OUTPUT_DIR = os.environ.get("OPTIMUS_CHECKPOINT_DIR")


def get_encoder(encoder_type="bert", output_encoder_dir="/tmp"):
    if not OUTPUT_DIR:
        raise Exception(
            "OPTIMUS_CHECKPOINT_DIR environment varialbe is required for running analogies. Please see example in src here."
        )
    checkpoint_encoder_dir = os.path.join(OUTPUT_DIR, "checkpoint-encoder-31250")
    encoder_config_class, encoder_model_class, encoder_tokenizer_class = MODEL_CLASSES[
        encoder_type
    ]
    model_encoder = encoder_model_class.from_pretrained(
        checkpoint_encoder_dir, latent_size=LATENT_SIZE_LARGE
    )
    tokenizer_encoder = encoder_tokenizer_class.from_pretrained(
        CODER_TYPE_TO_NAME[encoder_type], do_lower_case=True
    )
    model_encoder.to(get_device())
    return {"tokenizer": tokenizer_encoder, "model": model_encoder}


def get_decoder(decoder_type="gpt2", output_decoder_dir="/tmp"):
    if not OUTPUT_DIR:
        raise Exception(
            "OPTIMUS_CHECKPOINT_DIR environment varialbe is required for running analogies. Please see example in src here."
        )
    checkpoint_decoder_dir = os.path.join(OUTPUT_DIR, "checkpoint-decoder-31250")
    decoder_config_class, decoder_model_class, decoder_tokenizer_class = MODEL_CLASSES[
        decoder_type
    ]
    model_decoder = decoder_model_class.from_pretrained(
        checkpoint_decoder_dir, latent_size=LATENT_SIZE_LARGE
    )
    tokenizer_decoder = decoder_tokenizer_class.from_pretrained(
        CODER_TYPE_TO_NAME[decoder_type], do_lower_case=True
    )
    model_decoder.to(get_device())
    # Chunyuan: Add Padding token to GPT2
    special_tokens_dict = {
        "pad_token": "<PAD>",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>",
    }
    num_added_toks = tokenizer_decoder.add_special_tokens(special_tokens_dict)
    print("We have added", num_added_toks, "tokens to GPT2")
    model_decoder.resize_token_embeddings(
        len(tokenizer_decoder)
    )  # Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e. the length of the tokenizer.
    assert tokenizer_decoder.pad_token == "<PAD>"
    return {"tokenizer": tokenizer_decoder, "model": model_decoder}


def get_vae(model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, beta=1):
    if not OUTPUT_DIR:
        raise Exception(
            "OPTIMUS_CHECKPOINT_DIR environment varialbe is required for running analogies. Please see example in src here."
        )
    ArgsObj = namedtuple("Args", ["latent_size", "device", "fb_mode", "beta"])
    args = ArgsObj(
        latent_size=LATENT_SIZE_LARGE, device=get_device(), fb_mode=0, beta=beta
    )

    checkpoint_full_dir = os.path.join(OUTPUT_DIR, "checkpoint-full-31250")
    if not torch.cuda.is_available():
        checkpoint = torch.load(
            os.path.join(checkpoint_full_dir, "training.bin"), map_location="cpu"
        )
    else:
        checkpoint = torch.load(os.path.join(checkpoint_full_dir, "training.bin"))

    model_vae = VAE(
        model_encoder, model_decoder, tokenizer_encoder, tokenizer_decoder, args
    )
    model_vae.load_state_dict(checkpoint["model_state_dict"])
    # logger.info("Pre-trained Optimus is successfully loaded")
    model_vae.to(args.device)
    return model_vae


def eval_analogy(
    model_vae,
    tokenizer_encoder,
    tokenizer_decoder,
    a,
    b,
    c,
    degree_to_target=1.0,
    temperature=1.0,
    top_k=0,
    top_p=1.0,
):
    """
    Usage:
    r = get_encoder()
    s = get_decoder()
    v = v = get_vae(r['model'], s['model'], r['tokenizer'], s['tokenizer'])

    result = eval_analogy(v, r['tokenizer'], s['tokenizer'], 'I saw a truck', 'I saw an automobile', 'I saw a dog', temperature=0.01, degree_to_target=1)

    => i saw an animal


    """
    ArgsObj = namedtuple(
        "Args",
        [
            "degree_to_target",
            "device",
            "sent_source",
            "sent_target",
            "sent_input",
            "temperature",
            "top_k",
            "top_p",
        ],
    )
    args = ArgsObj(
        degree_to_target=degree_to_target,
        device=get_device(),
        sent_source=a,
        sent_target=b,
        sent_input=c,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )

    return runl.analogy(model_vae, tokenizer_encoder, tokenizer_decoder, args)
