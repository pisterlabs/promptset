#!/usr/bin/env python3
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models of the library (GPT/GPT-2/CTRL/Transformer-XL/XLNet)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import argparse
import logging
from tqdm import trange
import random

import torch
import torch.nn.functional as F
import numpy as np
import difflib

from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers import XLNetLMHeadModel, XLNetTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from transformers import CTRLLMHeadModel, CTRLTokenizer
from transformers import XLMWithLMHeadModel, XLMTokenizer


logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig)), ())

MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'ctrl': (CTRLLMHeadModel, CTRLTokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'xlnet': (XLNetLMHeadModel, XLNetTokenizer),
    'transfo-xl': (TransfoXLLMHeadModel, TransfoXLTokenizer),
    'xlm': (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PADDING_TEXT = """ In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing"""
#. <eod> </s> <eos>

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

def _unidiff_output(expected, actual):
    """
    Helper function. Returns a string containing the unified diff of two multiline strings.
    """

    expected=expected.splitlines(1)
    actual=actual.splitlines(1)

    diff=difflib.unified_diff(expected, actual)

    return ''.join(diff)

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits

def resample_sequence(model, length, context, resample_num=5, num_samples=1, temperature=0.5, repetition_penalty=1.0,
                       top_k=0, top_p=0.0, is_xlnet=False, device='cpu'):
    generated = context
    idxs = np.arange(generated.shape[1])[-length:]
    np.random.shuffle(idxs)
    with torch.no_grad():
        inputs = {'input_ids': generated}
        update_pos = idxs[:min(resample_num, len(idxs))]
        input_ids = generated
        perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]),
                             dtype=torch.float, device=device)
        perm_mask[:, :, update_pos] = 1.0


        target_mapping = torch.zeros((1, len(update_pos), input_ids.shape[1]), dtype=torch.float, device=device)
        target_mapping[0, torch.arange(len(update_pos)), update_pos] = 1.0

        inputs = {'input_ids': input_ids, 'perm_mask': perm_mask,  'target_mapping': target_mapping}
        outputs = model(**inputs)

        # next_token_logits = outputs[0][0, update_pos, :] / temperature
        next_token_logits = outputs[0][0,:,:] / temperature
        for _ in set(generated.view(-1).tolist()):
            next_token_logits[:,_] /= repetition_penalty
        # filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)

        generated[:, update_pos] = next_token[:, 0]

    return generated, update_pos

def random_permutation_sampling_sequential(model, length, context, num_samples=1, temperature=0.5, repetition_penalty=1.0,
                       top_k=0, top_p=0.0, is_xlnet=False, device='cpu'):

    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context #maybe add a for loop later
    input_ids = torch.cat((generated, torch.zeros((1, length), dtype=torch.long, device=device)), dim=1)

    with torch.no_grad():
        random_permutation = np.arange(input_ids.shape[1])[-length:]
        np.random.shuffle(random_permutation)
        perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)
        perm_mask[:, :, -length:] = 1.0
        for idx, perm_idx in enumerate(random_permutation):
            perm_mask[:,:, random_permutation[:idx]] = 0.

            target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float, device=device)
            target_mapping[0,0, perm_idx] = 1.0
            inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping}
            outputs = model(**inputs)
            next_token_logits = outputs[0][0,:,:] / temperature
            for _ in set(generated.view(-1).tolist()):
                next_token_logits[:,_] /= repetition_penalty
        # filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)

            input_ids[:, perm_idx] = next_token[:, 0]

    return input_ids, random_permutation


def random_permutation_sampling(model, length, context, num_samples=1, temperature=0.5, repetition_penalty=1.0,
                       top_k=0, top_p=0.0, is_xlnet=False, device='cpu'):

    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context #maybe add a for loop later
    input_ids = torch.cat((generated, torch.zeros((1, length), dtype=torch.long, device=device)), dim=1)

    with torch.no_grad():
        random_permutation = np.arange(input_ids.shape[1])[-length:]
        np.random.shuffle(random_permutation)
        perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)
        perm_mask[:, :, -length:] = 1.0  # Previous tokens don't see last token
        target_mapping = torch.zeros((1, length, input_ids.shape[1]), dtype=torch.float, device=device)
        print(random_permutation)
        for idx, perm_idx in enumerate(random_permutation):
            perm_mask[:, perm_idx, random_permutation[:idx]] = 0.

        print(perm_mask[:, -length:, -length:])
        target_mapping[0, torch.arange(length), random_permutation] = 1.0
        # print(target_mapping[0, -length:,:])
        inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping}
        outputs = model(**inputs)
        # next_token_logits = outputs[0][0, update_pos, :] / temperature
        next_token_logits = outputs[0][0,:,:] / temperature
        for _ in set(generated.view(-1).tolist()):
            next_token_logits[:,_] /= repetition_penalty
        # filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
        print(random_permutation)
        input_ids[:, random_permutation] = next_token[:, 0]

    return input_ids, random_permutation

def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,
                    is_xlnet=False, is_xlm_mlm=False, xlm_mask_token=None, xlm_lang=None, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in range(length):

            inputs = {'input_ids': generated}
            if is_xlnet:
                # XLNet is a direct (predict same token, not next token) and bi-directional model by default
                # => need one additional dummy token in the input (will be masked), attention mask and target mapping (see model docstring)
                input_ids = torch.cat((generated, torch.zeros((1, 1), dtype=torch.long, device=device)), dim=1)
                perm_mask = torch.zeros((1, input_ids.shape[1], input_ids.shape[1]), dtype=torch.float, device=device)
                perm_mask[:, :, -1] = 1.0  # Previous tokens don't see last token
                target_mapping = torch.zeros((1, 1, input_ids.shape[1]), dtype=torch.float, device=device)
                target_mapping[0, 0, -1] = 1.0  # predict last token
                inputs = {'input_ids': input_ids, 'perm_mask': perm_mask, 'target_mapping': target_mapping}

            if is_xlm_mlm and xlm_mask_token:
                # XLM MLM models are direct models (predict same token, not next token)
                # => need one additional dummy token in the input (will be masked and guessed)
                input_ids = torch.cat((generated, torch.full((1, 1), xlm_mask_token, dtype=torch.long, device=device)), dim=1)
                inputs = {'input_ids': input_ids}

            if xlm_lang is not None:
                inputs["langs"] = torch.tensor([xlm_lang] * inputs["input_ids"].shape[1], device=device).view(1, -1)

            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / (temperature if temperature > 0 else 1.)

            # reptition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for _ in set(generated.view(-1).tolist()):
                next_token_logits[_] /= repetition_penalty

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: #greedy sampling:
                next_token = torch.argmax(filtered_logits).unsqueeze(0)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--mode", type=str, default="ltr")
    parser.add_argument("--refine", type=str, default='none', choices=['none', 'gibbs', 'xent'])
    parser.add_argument("--padding_src", type=str, default="../data/bc_50k.txt")
    parser.add_argument("--xlm_lang", type=str, default="", help="Optional language when used with the XLM model.")
    parser.add_argument("--length", type=int, default=20)
    parser.add_argument("--resample_num", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--stop_token', type=str, default=None,
                        help="Token at which text generation is stopped")
    parser.add_argument('--num_gen', type=int, default=10)
    parser.add_argument('--out_file', type=str, default="")
    args = parser.parse_args()

    if not args.out_file:
        args.out_file = "generation_{}_{}.txt".format(args.mode, args.length)

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    if args.length < 0 and model.config.max_position_embeddings > 0:
        args.length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.length:
        args.length = model.config.max_position_embeddings  # No generation bigger than model size
    elif args.length < 0:
        args.length = MAX_LENGTH  # avoid infinite loop

    logger.info(args)
    if args.model_type in ["ctrl"]:
        if args.temperature > 0.7 :
            logger.info('CTRL typically works better with lower temperatures (and lower top_k).')
    with open(args.out_file, 'w') as outfile:
        for gen_id in range(args.num_gen):
            xlm_lang = None
            # XLM Language usage detailed in the issues #1414
            if args.model_type in ["xlm"] and hasattr(tokenizer, 'lang2id') and hasattr(model.config, 'use_lang_emb') \
                    and model.config.use_lang_emb:
                if args.xlm_lang:
                    language = args.xlm_lang
                else:
                    language = None
                    while language not in tokenizer.lang2id.keys():
                        language = input("Using XLM. Select language in " + str(list(tokenizer.lang2id.keys())) + " >>> ")
                xlm_lang = tokenizer.lang2id[language]

            is_xlm_mlm = args.model_type in ["xlm"] and 'mlm' in args.model_name_or_path
            if is_xlm_mlm:
                xlm_mask_token = tokenizer.mask_token_id
            else:
                xlm_mask_token = None

            raw_text = "" #args.prompt if args.prompt else input("Model prompt >>> ")
            if args.model_type in ["transfo-xl", "xlnet"]:
                # Models with memory likes to have a long prompt for short inputs.
                with open(args.padding_src, 'r') as padding_src:
                    lines = padding_src.readlines()
                    random_index = random.randint(0, len(lines) - 10)
                    PADDING_TEXT = " ".join(lines[random_index: random_index+10])

                raw_text = (args.padding_text if args.padding_text else PADDING_TEXT) + raw_text
                print("Seed: \n {} \n -------".format(raw_text))
            outfile.write(raw_text)
            outfile.write('\n')
            seed_len = len(raw_text)
            context_tokens = tokenizer.encode(raw_text)
            if args.mode == "ltr":
                context_out = sample_sequence(
                    model=model,
                    context=context_tokens,
                    length=args.length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    is_xlnet=bool(args.model_type == "xlnet"),
                    is_xlm_mlm=is_xlm_mlm,
                    xlm_mask_token=xlm_mask_token,
                    xlm_lang=xlm_lang,
                    device=args.device,
                )
            elif args.mode == "rnd":
                context_out, random_permutation = random_permutation_sampling_sequential(
                    model=model,
                    context=context_tokens,
                    length=args.length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    device=args.device,
                )


            out = context_out[0, len(context_tokens):].tolist()
            raw_text_extended = raw_text + " " + bcolors.OKBLUE +tokenizer.decode(out, clean_up_tokenization_spaces=True) + bcolors.ENDC
            print("Extended: \n {} \n -------".format(raw_text_extended))
            raw_text = raw_text + " " + tokenizer.decode(out, clean_up_tokenization_spaces=True)

            before_resample = raw_text
            if args.refine == "gibbs":
                context_tokens = tokenizer.encode(raw_text)
                context = torch.tensor(context_tokens, dtype=torch.long, device=args.device)
                context = context.unsqueeze(0).repeat(1, 1)
                for _ in range(50):
                    context_out, update_pos = resample_sequence(
                        model=model,
                        context=context,
                        length=args.length,
                        resample_num=args.resample_num,
                        temperature=args.temperature,
                        repetition_penalty=args.repetition_penalty,
                        top_k=args.top_k,
                        top_p=args.top_p,
                        device=args.device,
                        is_xlnet=bool(args.model_type == "xlnet"),
                    )
                    out = context_out[0, :].tolist()
                    tokens = tokenizer.convert_ids_to_tokens(out) #, clean_up_tokenization_spaces=True)
                    for i in update_pos:
                        tokens[i] = bcolors.OKGREEN + tokens[i] + bcolors.ENDC
                    text = tokenizer.convert_tokens_to_string(tokens)
                    # print(text)
                    # print ("-----------")

                print("Refined: \n {} \n -------".format(text))
                raw_text = text.translate({ord("]"):'', ord("["):''})
            # print(_unidiff_output(before_resample, after_resample))
            outfile.write(raw_text[seed_len:])
            outfile.write("\n-\n")

    print ("Done :)")


if __name__ == '__main__':
    main()
