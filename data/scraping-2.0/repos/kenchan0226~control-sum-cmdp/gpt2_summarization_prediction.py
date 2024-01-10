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
from tqdm import trange, tqdm

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler
from cytoolz import curry
from os.path import join
from utils import io
import nltk
import os
import json
from utils.io import LEN_BINS, ext_frag_density_to_bin, n_gram_novelty_to_bin, fusion_ratio_to_bin

from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers import XLNetLMHeadModel, XLNetTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from transformers import CTRLLMHeadModel, CTRLTokenizer
from transformers import XLMWithLMHeadModel, XLMTokenizer
from gpt2_summarization_finetuning import SummarizationDataset, get_control_mode_special_ids_dict

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in
                  (GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig)), ())

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
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (batch size x vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
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

        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(dim=1, index=sorted_indices, src=sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
    return logits


def sample_sequence(model, length, context, num_samples=1, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,
                    is_xlnet=False, is_xlm_mlm=False, xlm_mask_token=None, xlm_lang=None, device='cpu'):
    context = torch.tensor(context, dtype=torch.long, device=device)
    context = context.unsqueeze(0).repeat(num_samples, 1)
    generated = context
    with torch.no_grad():
        for _ in trange(length):

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
                input_ids = torch.cat((generated, torch.full((1, 1), xlm_mask_token, dtype=torch.long, device=device)),
                                      dim=1)
                inputs = {'input_ids': input_ids}

            if xlm_lang is not None:
                inputs["langs"] = torch.tensor([xlm_lang] * inputs["input_ids"].shape[1], device=device).view(1, -1)

            outputs = model(
                **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0:  # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)
    return generated


def load_and_cache_examples(args, tokenizer, split):
    dataset = SummarizationDataset(tokenizer, args, data_dir=args.data_dir, split=split, control_modes=args.control_modes, is_multi_reference=args.multiple_reference)
    return dataset


@curry
def coll(batch, TLDR_id_list, pad_idx, tokenizer, control_modes=[], with_ground_truth=False, desired_target_numbers=[], control_mode_special_ids_dict={}, input_trunc_len=512):
    #doc_seq_lens = []
    #for doc, _, _, _, _ in batch:
    #    doc_seq_lens.append( len(doc[:input_trunc_len]) )
    #max_doc_len = max(doc_seq_lens)
    #max_len += len(control_modes)  # the token of each control mode consumes one position
    #if 2 in control_modes:  # for exact length control, a number may have three tokens
    #    max_len += 2
    input_lens = []
    doc_trunc_ids_all = []
    prefix_ids_all = []
    #max_number_length_token_ids = 3

    for doc, _, _, summary_sent_list, controllable_fields in batch:
        # process str
        summary_str = ' '.join(summary_sent_list)
        summary_word_list = summary_str.split(' ')
        summary_len = len(summary_word_list)
        # process tensors
        doc_truncated = doc[:input_trunc_len]
        # control mode specific operations
        special_token_id_list = []
        for control_mode, desired_target_number in zip(control_modes, desired_target_numbers):
            #print(control_mode)
            if control_mode == 1:
                if with_ground_truth:
                    len_bin = LEN_BINS[summary_len]
                else:
                    len_bin = desired_target_number
                #print(len_bin)
                #print(len_bin_idx)
                special_token_id_list.append(control_mode_special_ids_dict['len_bin'][len_bin])
                # len_bin_list.append(len_bin)
            elif control_mode == 2:
                if with_ground_truth:
                    target_len = summary_len
                else:
                    target_len = desired_target_number
                length_token_ids = tokenizer.convert_tokens_to_ids([str(target_len)])
                special_token_id_list += length_token_ids
                #number_length_token_ids = len(length_token_ids)
                #if number_length_token_ids > max_number_length_token_ids:
                #    max_number_length_token_ids = number_length_token_ids
            elif control_mode == 5:
                if with_ground_truth:
                    abs_bin = ext_frag_density_to_bin(controllable_fields['ext_frag_density'])
                else:
                    abs_bin = desired_target_number
                special_token_id_list.append(control_mode_special_ids_dict['abs_bin'][abs_bin])
                #doc_truncated.insert(0, abs_bin_idx)
                #abs_bin_list.append(abs_bin)
            elif control_mode == 4:
                if with_ground_truth:
                    abs_bin = n_gram_novelty_to_bin(controllable_fields['two_gram_novelty'])
                else:
                    abs_bin = desired_target_number
                special_token_id_list.append(control_mode_special_ids_dict['abs_bin'][abs_bin])
                #abs_bin_list.append(abs_bin)
            elif control_mode == 6:
                if with_ground_truth:
                    abs_bin = fusion_ratio_to_bin(controllable_fields['avg_fusion_ratio'])
                else:
                    abs_bin = desired_target_number
                special_token_id_list.append(control_mode_special_ids_dict['abs_bin'][abs_bin])
                #abs_bin_list.append(abs_bin)
            elif control_mode == 7:
                if with_ground_truth:
                    doc_truncated = controllable_fields['reference_entities_prefix_ids'] + doc_truncated
                    #special_token_id_list += controllable_fields['reference_entities_prefix_ids']
                else:
                    raise ValueError

        doc_trunc_ids_all.append(doc_truncated)
        prefix_ids = TLDR_id_list[:-1] + special_token_id_list + TLDR_id_list[-1:]
        prefix_ids_all.append(prefix_ids)
        input_lens.append( len(doc_truncated) + len(prefix_ids) )

    """
    print(tokenizer.decode(doc_trunc_ids_all[0], clean_up_tokenization_spaces=False))
    print()
    print(tokenizer.decode(doc_trunc_ids_all[1], clean_up_tokenization_spaces=False))
    print()
    print(tokenizer.decode(doc_trunc_ids_all[2], clean_up_tokenization_spaces=False))
    print()
    print(tokenizer.decode(prefix_ids_all[0], clean_up_tokenization_spaces=False))
    print()
    print(tokenizer.decode(prefix_ids_all[1], clean_up_tokenization_spaces=False))
    print()
    print(tokenizer.decode(prefix_ids_all[2], clean_up_tokenization_spaces=False))
    print()
    """

    max_input_len = max(input_lens)
    input_ids_all_padded = []
    position_ids_all_padded = []
    for doc_trunc_ids, prefix_ids in zip(doc_trunc_ids_all, prefix_ids_all):
        doc_len = len(doc_trunc_ids)
        padding_length = max_input_len - doc_len - len(prefix_ids)
        input_ids = doc_trunc_ids + ([pad_idx] * padding_length) + prefix_ids
        position_ids = list(range(doc_len + padding_length)) + list(range(doc_len, doc_len + len(prefix_ids)))
        input_ids_all_padded.append(input_ids)
        position_ids_all_padded.append(position_ids)

    """
    print(input_ids_all_padded[0])
    print()
    print(input_ids_all_padded[1])
    print()
    print(input_ids_all_padded[2])
    print()

    print(tokenizer.decode(input_ids_all_padded[0], clean_up_tokenization_spaces=False))
    print()
    print(tokenizer.decode(input_ids_all_padded[1], clean_up_tokenization_spaces=False))
    print()
    print(tokenizer.decode(input_ids_all_padded[2], clean_up_tokenization_spaces=False))
    print()
    exit()
    """
    

    """
    doc_len = len(doc_truncated)
    padding_length = max_doc_len - doc_len
    if 2 in control_modes:
        padding_length += (max_number_length_token_ids - number_length_token_ids)
        #padded_doc_ids += [pad_idx] * (max_number_length_token_ids - number_length_token_ids)
    input_ids = doc_truncated + ([pad_idx] * padding_length) + TLDR_id_list[:-1] + special_token_id_list + TLDR_id_list[-1:]
    position_ids = list(range( doc_len + padding_length )) + list(range(doc_len, doc_len + len(TLDR_id_list) + len(special_token_id_list) ))
    input_ids_all.append(input_ids)
    position_ids_all.append(position_ids)
    """

    input_ids_tensor = torch.LongTensor(input_ids_all_padded)
    position_ids_tensor = torch.LongTensor(position_ids_all_padded)
    return input_ids_tensor, position_ids_tensor


def predict(args, model, max_output_length, tokenizer, split, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0,
            control_modes=[], with_ground_truth_input=False, desired_target_numbers=[], control_mode_special_ids_dict={}):
    predict_dataset = load_and_cache_examples(args, tokenizer, split=split)
    predict_sampler = SequentialSampler(predict_dataset)
    #TLDR_id_list = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<control><summarize>:"))
    TLDR_id_list = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("TL;DR:"))
    pad_idx = tokenizer.convert_tokens_to_ids(['<pad>'])[0]
    eos_idx = tokenizer.eos_token_id
    num_exported_samples = 0
    predict_dataloader = DataLoader(predict_dataset, sampler=predict_sampler, batch_size=args.batch_size,
                                 collate_fn=coll(TLDR_id_list=TLDR_id_list, pad_idx=pad_idx, tokenizer=tokenizer, control_modes=control_modes,
                                                 with_ground_truth=with_ground_truth_input, desired_target_numbers=desired_target_numbers,
                                                 control_mode_special_ids_dict=control_mode_special_ids_dict,
                                                 input_trunc_len=args.input_trunc_length))
    model.eval()

    for batch_i, batch in enumerate(tqdm(predict_dataloader)):
        context, position_ids = batch
        batch_size = context.size(0)
        # attn_mask
        mask = torch.ne(context, pad_idx).float()
        context = context.to(args.device)
        position_ids = position_ids.to(args.device)
        mask = mask.to(args.device)

        input_ids = context
        input_sequence_length = input_ids.size(1)

        with torch.no_grad():
            outputs_tensor = model.generate(input_ids=input_ids, max_length=max_output_length + input_sequence_length, do_sample=args.do_sample,
                                       num_beams=args.beam_size, temperature=args.temperature, top_k=args.top_k,
                                       top_p=args.top_p, repetition_penalty=args.repetition_penalty, pad_token_id=pad_idx,
                                       eos_token_ids=[eos_idx], num_return_sequences=1,
                                       position_ids=position_ids, attn_mask=mask)  # tensor:  [batch, max_out_seq_len]

        if len(outputs_tensor.shape) > 2:
            outputs_tensor = outputs_tensor.squeeze()

        # remove the input ids from output tensor
        outputs_tensor = outputs_tensor[:, input_sequence_length:]

        outputs = outputs_tensor.tolist()
        for out_ids in outputs:
            eos_positions = [position for position, word_id in enumerate(out_ids) if word_id == eos_idx]
            if len(eos_positions) > 0:
                end_position = eos_positions[0]
                out_ids = out_ids[:end_position]
            out_text = tokenizer.decode(out_ids, clean_up_tokenization_spaces=False)
            decode_out_sent_list = nltk.tokenize.sent_tokenize(out_text)
            # output the predicted sentences to a file
            with open(join(args.pred_path, 'output/{}.dec'.format(num_exported_samples)), 'w') as f:
                f.write(io.make_html_safe('\n'.join(decode_out_sent_list)))
            num_exported_samples += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(
                            ALL_MODELS))
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--pred_path", default=None, type=str, required=True,
                        help="The path of output dir.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The path of the directory containing all the splits.")
    parser.add_argument("--split", default=None, type=str, required=True,
                        help="The split to be predicted.")
    #parser.add_argument("--prompt", type=str, default="")
    #parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--xlm_lang", type=str, default="", help="Optional language when used with the XLM model.")
    parser.add_argument("--max_output_length", type=int, default=120)
    parser.add_argument('--input_trunc_length', type=int, default=512,
                        help='Max length of output.')
    #parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.0,
                        help="primarily useful for CTRL model; in that case, use 1.2")
    parser.add_argument("--beam_size", type=int, default=1,
                        help="Beam size")
    parser.add_argument("--do_sample", action="store_true", default=False,
                        help="Do sampling or not")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--multiple_reference", action="store_true", default=False, help="for duc2002")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--control_modes', nargs='+', default=[], type=int,
                        help='0: nothing. 1: control length bin. 2: control exact length. 3: novel 3-gram range. 4: novel 2-gram range. 5: extractive_fragment density bin. 6: sentence fusion')
    parser.add_argument('--with_ground_truth_input', action="store_true", default=False,
                        help='Provide the ground-truth len bin.')
    parser.add_argument('--desired_target_numbers', nargs='+', default=[], type=int,
                        help='The target len bin.')
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    #parser.add_argument('--stop_token', type=str, default=None,
    #                    help="Token at which text generation is stopped")
    args = parser.parse_args()

    os.makedirs(args.pred_path)
    os.makedirs(join(args.pred_path, 'output'))

    json.dump(vars(args), open(join(args.pred_path, 'log.json'), 'w'))

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 1
    args.overwrite_cache = False
    #args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.model_type = args.model_type.lower()

    if args.with_ground_truth_input:
        args.desired_target_numbers = [-1] * len(args.control_modes)

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

    # get ids of special tokens for different control modes
    control_mode_special_ids_dict = get_control_mode_special_ids_dict(args.control_modes, tokenizer)
    print()
    print(control_mode_special_ids_dict)
    #print(tokenizer.convert_tokens_to_ids(["<control>", "<summarize>", "<ent>", "<start>"]))
    #print(["Marseille", "<ent>", "Bild", "<ent>", "Paris Match"])
    #s = "Marseille Bild Paris Match"
    #tokens = tokenizer.tokenize(s)
    #print(tokens)
    #print(tokenizer.convert_tokens_to_ids(tokens))
    #print(tokenizer.decode(tokenizer.convert_tokens_to_ids(tokens), clean_up_tokenization_spaces=False))
    #print()

    if args.max_output_length < 0 and model.config.max_position_embeddings > 0:
        args.max_output_length = model.config.max_position_embeddings
    elif 0 < model.config.max_position_embeddings < args.max_output_length:
        args.max_output_length = model.config.max_position_embeddings  # No generation bigger than model size
    elif args.max_output_length < 0:
        args.max_output_length = MAX_LENGTH  # avoid infinite loop

    logger.info(args)
    if args.model_type in ["ctrl"]:
        if args.temperature > 0.7:
            logger.info('CTRL typically works better with lower temperatures (and lower top_k).')

    predict(args, model, args.max_output_length, tokenizer, args.split, args.temperature, args.top_k, args.top_p,
            args.repetition_penalty, args.control_modes, args.with_ground_truth_input, args.desired_target_numbers, control_mode_special_ids_dict)

if __name__ == '__main__':
    main()
