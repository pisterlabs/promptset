# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat

import torch
import torch.nn.functional as F

# from pytorch_pretrained_bert import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from pytorch_pretrained_bert import OpenAIGPTTokenizer, GPT2Tokenizer
from modeling_openai import OpenAIGPTLMHeadModel
from modeling_gpt2 import GPT2LMHeadModel
# from train import SPECIAL_TOKENS, build_input_from_segments
from utils import SPECIAL_TOKENS
from utils import get_dataset_personalities, download_pretrained_model

import numpy as np
np.set_printoptions(threshold=np.inf)


def top_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k, top-p (nucleus) and/or threshold filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k: <=0: no filtering, >0: keep only top k tokens with highest probability.
            top_p: <=0.0: no filtering, >0.0: keep only a subset S of candidates, where S is the smallest subset
                whose total probability mass is greater than or equal to the threshold top_p.
                In practice, we select the highest probability tokens whose cumulative probability mass exceeds
                the threshold top_p.
            threshold: a minimal threshold to keep logits
    """
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate a bit the code
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        # Remove all tokens with a probability less than the last token in the top-k tokens
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # Compute cumulative probabilities of sorted tokens
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probabilities = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probabilities > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        # Back to unsorted indices and set them to -infinity
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value

    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value

    return logits


def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    # YW: speaker1 is user
    instance = {}
    sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]   # YW: add speaker infomation

    instance["input_ids"] = list(chain(*sequence))   # YW: concat all the context
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]   # YW: TODO: persona is speaker1?
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    instance["lm_labels"] = [-1] * len(instance["input_ids"])   # YW: all -1 (mask?) -> because it's not the right candidate(label)!
    if lm_labels:   # YW: if it's label
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
    return instance, sequence


def sample_sequence(personality, history, tokenizer, model, args, current_output=None):
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []

    for i in range(args.max_length):
        instance, sequence = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=args.device).unsqueeze(0)
        token_type_ids = torch.tensor(instance["token_type_ids"], device=args.device).unsqueeze(0)

        logits = model(input_ids, mc_token_ids=-1, token_type_ids=token_type_ids, gen=True)

        if "gpt2" == args.model:
            logits = logits[0]
        logits = logits[0, -1, :] / args.temperature

        logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
        # logits_arr = logits.cpu().detach().numpy()
        probs = F.softmax(logits, dim=-1)
        # probs_arr = probs.cpu().detach().numpy()
        # non_zero_indices = np.nonzero(probs_arr)
        # non_zero_probs = [probs_arr[index] for index in non_zero_indices]

        prev = torch.topk(probs, 1)[1] if args.no_sample else torch.multinomial(probs, 1)
        if i < args.min_length and prev.item() in special_tokens_ids:
            print('INFO - before while - prev.item()', prev.item())
            j = 0
            while prev.item() in special_tokens_ids and j < args.max_length:
                print('INFO - 105 - in while')
                prev = torch.multinomial(probs, num_samples=1)
                print('INFO - in while - prev.item()', prev.item())
                j += 1
            else:
                current_output.append(prev.item())

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output


def generate(persona, history, tokenizer, model, args):
    with torch.no_grad():
        out_ids = sample_sequence(persona, history, tokenizer, model, args)
    out_text = tokenizer.decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return out_text, out_ids
