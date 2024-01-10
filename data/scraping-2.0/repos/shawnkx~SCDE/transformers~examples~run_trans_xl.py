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

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import json

from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers import XLNetLMHeadModel, XLNetTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from transformers import CTRLLMHeadModel, CTRLTokenizer
from transformers import XLMWithLMHeadModel, XLMTokenizer

from itertools import permutations 
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
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


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
                input_ids = torch.cat((generated, torch.full((1, 1), xlm_mask_token, dtype=torch.long, device=device)), dim=1)
                inputs = {'input_ids': input_ids}

            if xlm_lang is not None:
                inputs["langs"] = torch.tensor([xlm_lang] * inputs["input_ids"].shape[1], device=device).view(1, -1)

            outputs = model(**inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet/CTRL (cached hidden-states)
            next_token_logits = outputs[0][0, -1, :] / (temperature if temperature > 0 else 1.)

            # reptition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for _ in set(generated):
                next_token_logits[_] /= repetition_penalty
                
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: #greedy sampling:
                next_token = torch.argmax(filtered_logits).unsqueeze(0)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
    return generated


def main(inc=True):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--xlm_lang", type=str, default="", help="Optional language when used with the XLM model.")
    parser.add_argument("--length", type=int, default=20)
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
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.model_type = args.model_type.lower()
    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    #model = torch.nn.DataParallel(model)

    model.to(args.device)
    
    model.eval()
    device = args.device
    criterion = nn.CrossEntropyLoss()
    max_sen = 3
    # if args.length < 0 and model.config.max_position_embeddings > 0:
    #     args.length = model.config.max_position_embeddings
    # elif 0 < model.config.max_position_embeddings < args.length:
    #     args.length = model.config.max_position_embeddings  # No generation bigger than model size 
    # elif args.length < 0:
    #     args.length = MAX_LENGTH  # avoid infinite loop

    # logger.info(args)
    # if args.model_type in ["ctrl"]:
    #     if args.temperature > 0.7 : 
    #         logger.info('CTRL typically works better with lower temperatures (and lower top_k).')

    # EXH
    if not inc:
        with open('../../final_data_exp/data/test_remove_distractor.json') as f:
            datas = json.load(f)
            correct = 0.
            total = 0
            whole_correct = 0.
            whole_total = 0
            for idx, data in enumerate(datas):
                print(idx)
                #single_answer = [num_candidates]
                eid = data['eid']

                passage = data['passage']
                candidates = data['candidates']
                answers = data['answer_sequence']
                num_blanks = data['number_of_blanks']
                num_candidates = data['candidate_length']
                #print(passage, answers, candidates, idx)
                cur_prob = float('inf')
                cur_rst = None
                blank_indexes = [passage.index('<' + str(i + 1) + '>') for i in range(num_blanks)]
                acc_scores = []
                golden_ans = [ans[1] for ans in answers]
                #dis = [ans for ans in list(range(num_candidates)) if ans not in golden_ans]
                for idx, bidx in enumerate(blank_indexes):
                    acc_scores.append([])
                    left_context = ' '.join(passage[bidx-max_sen:bidx])
                    right_context = ' '.join(passage[bidx+1:bidx+1+max_sen])
                    for i in range(num_candidates):
                        cur_context = ' '.join((left_context, candidates[i], right_context))
                        cur_context= tokenizer.encode(cur_context)
                        cur_context = torch.tensor(cur_context, dtype=torch.long, device=device)
                        cur_context = cur_context.unsqueeze(0)
                        inp = cur_context[:,:-1]
                        out = cur_context[:, 1:]
                        pred = model(inp)[0]
                        loss = criterion(pred.view(-1, pred.shape[2]), out.view(-1)).item()
                        #print(len(acc_scores), idx)
                        acc_scores[idx].append(loss)
               
                for perm in permutations(list(range(num_candidates)), len(answers)):
                    cur_loss = 0.
                    # flag = False
                    # for d in dis:
                    #     if d in perm:
                    #         flag = True
                    #         break
                    # if flag:
                    #     continue
                    for idx, p in enumerate(perm):
                        #print(idx, p)
                        cur_loss += acc_scores[idx][p]

                    # for idx, bidx in enumerate(blank_indexes):
                    #     passage[bidx] = candidates[perm[idx]]
                    # cur_loss = 0.
                    # for idx, bidx in enumerate(blank_indexes):
                    #     left_context = ' '.join(passage[bidx-max_sen:bidx])
                    #     right_context = ' '.join(passage[bidx+1:bidx+1+max_sen])
                    #     cur_context = ' '.join((left_context, passage[bidx], right_context))
                    #     #print(cur_context)
                    #     #exit()
                    #     cur_context= tokenizer.encode(cur_context)
                    #     cur_context = torch.tensor(cur_context, dtype=torch.long, device=device)
                    #     cur_context = cur_context.unsqueeze(0)
                    #     inp = cur_context[:,:-1]
                    #     out = cur_context[:, 1:]
                    #     pred = model(inp)[0]
                    #     loss = criterion(pred.view(-1, pred.shape[2]), out.view(-1)).item()
                    #     cur_loss += loss
                    #print(cur_rst)
                    
                    #print(cur_loss, perm)
                    if cur_loss < cur_prob:
                        #print(cur_loss, 'xiang', perm)
                        cur_prob = cur_loss
                        cur_rst = perm
                print(cur_rst)
                for l, r in zip(cur_rst, answers):
                    if l == r[1]:
                        correct += 1
                    total += 1
                golden = [an[1] for an in answers]
                print(cur_rst, golden)
                if list(cur_rst) == golden:
                    whole_correct += 1
                whole_total += 1
                print(correct / total, whole_correct / whole_total)

    # INC
    else:
        print('use inc to decode')
        with open('../../final_data_exp/data/test.json') as f:
            datas = json.load(f)
            correct = 0.
            total = 0
            whole_correct = 0.
            whole_total = 0
            for idx, data in enumerate(datas):
                
                #single_answer = [num_candidates]
                eid = data['eid']
                print(eid)
                passage = data['passage']
                candidates = data['candidates']
                answers = data['answer_sequence']
                num_blanks = data['number_of_blanks']
                num_candidates = data['candidate_length']
                #print(passage, answers, candidates, idx)
                cur_prob = float('inf')
                cur_rst = None
                blank_indexes = [passage.index('<' + str(i + 1) + '>') for i in range(num_blanks)]
                candi_list = list(range(num_candidates))
                #context = ''.join(passage[:blank_indexes[0]])
                rst = []
                for idx, bidx in enumerate(blank_indexes):
                    context = ' '.join(passage[:bidx])
                    cur_loss = float('inf')
                    left_context = ' '.join(passage[bidx-max_sen:bidx])
                    right_context = ' '.join(passage[bidx+1:bidx+1+max_sen])
                    for can_idx in candi_list:
                        cur_context = ' '.join((left_context, candidates[can_idx], right_context))
                        #print(cur_context)
                        #exit()
                        cur_context= tokenizer.encode(cur_context)
                        cur_context = torch.tensor(cur_context, dtype=torch.long, device=device)
                        cur_context = cur_context.unsqueeze(0)
                        inp = cur_context[:,:-1]
                        out = cur_context[:, 1:]
                        pred = model(inp)[0]
                        loss = criterion(pred.view(-1, pred.shape[2]), out.view(-1)).item()
                        if loss < cur_loss:
                            cur_loss = loss
                            cur_rst = can_idx
                    passage[bidx] = candidates[cur_rst]
                    candi_list.remove(cur_rst)
                    rst.append(cur_rst)
                for l, r in zip(rst, answers):
                    if l == r[1]:
                        correct += 1
                    total += 1
                golden = [an[1] for an in answers]
                if rst == golden:
                    whole_correct += 1
                whole_total += 1
                print(correct / total, whole_correct / whole_total)



if __name__ == '__main__':
    main(False)
