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
import re
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler, Dataset
from cytoolz import curry
from os.path import join
import nltk
import os
import json

from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers import XLNetLMHeadModel, XLNetTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from transformers import CTRLLMHeadModel, CTRLTokenizer
from transformers import XLMWithLMHeadModel, XLMTokenizer
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop


MODEL_CLASSES = {
    'gpt2': (GPT2LMHeadModel, GPT2Tokenizer),
    'ctrl': (CTRLLMHeadModel, CTRLTokenizer),
    'openai-gpt': (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'xlnet': (XLNetLMHeadModel, XLNetTokenizer),
    'transfo-xl': (TransfoXLLMHeadModel, TransfoXLTokenizer),
    'xlm': (XLMWithLMHeadModel, XLMTokenizer),
    'bart': (BartForConditionalGeneration, BartTokenizer),
    'pegasus': (PegasusForConditionalGeneration, PegasusTokenizer)
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


def _count_data(path):
    """ count number of data in the given path"""
    matcher = re.compile(r'[0-9]+\.json')
    match = lambda name: bool(matcher.match(name))
    names = os.listdir(path)
    n_data = len(list(filter(match, names)))
    return n_data


def make_html_safe(s):
    """Rouge use html, has to make output html safe"""
    return s.replace("<", "&lt;").replace(">", "&gt;")


class TestDataset(Dataset):
    def __init__(self, data_dir, split):
        assert split in ['test']
        self._data_path = join(data_dir, split)
        self._n_data = _count_data(self._data_path)

    def __len__(self) -> int:
        return self._n_data

    def __getitem__(self, i: int):
        with open(join(self._data_path, '{}.json'.format(i))) as f:
            js = json.loads(f.read())
        return " ".join(js["article"])


def load_and_cache_examples(args, tokenizer, split):
    dataset = TestDataset(data_dir=args.data_dir, split=split)
    return dataset


@curry
def coll(batch, tokenizer, input_trunc_len=512):
    src_text = []
    for doc_str in batch:
        src_text.append(doc_str)
    batch = tokenizer.prepare_seq2seq_batch(src_text, truncation=True, padding='longest', return_tensors="pt", max_length=input_trunc_len)

    return batch.input_ids, batch.attention_mask


def predict(args, model, max_output_length, tokenizer, split, temperature=1, top_k=0, top_p=0.0, repetition_penalty=1.0):
    predict_dataset = load_and_cache_examples(args, tokenizer, split=split)
    predict_sampler = SequentialSampler(predict_dataset)
    # TLDR_id_list = tokenizer.convert_tokens_to_ids(tokenizer.tokenize("<control><summarize>:"))
    pad_idx = tokenizer.pad_token_id
    #assert pad_idx == 0
    eos_idx = tokenizer.eos_token_id
    num_exported_samples = 0
    predict_dataloader = DataLoader(predict_dataset, sampler=predict_sampler, batch_size=args.batch_size,
                                    collate_fn=coll(tokenizer=tokenizer,
                                                    input_trunc_len=args.input_trunc_length))
    model.eval()

    for batch_i, batch in enumerate(tqdm(predict_dataloader)):
        context, attention_mask = batch
        # attn_mask
        context = context.to(args.device)
        attention_mask = attention_mask.to(args.device)

        input_ids = context
        input_sequence_length = input_ids.size(1)

        #print("input_ids size")
        #print(input_ids.size())
        #print("mask size")
        #print(attention_mask.size())

        with torch.no_grad():
            outputs_tensor = model.generate(input_ids=input_ids, max_length=max_output_length,
                                            do_sample=args.do_sample,
                                            num_beams=args.beam_size, temperature=args.temperature, top_k=args.top_k,
                                            top_p=args.top_p, repetition_penalty=args.repetition_penalty,
                                            pad_token_id=pad_idx, num_return_sequences=1,
                                            attention_mask=attention_mask)  # tensor:  [batch, max_out_seq_len]

        if len(outputs_tensor.shape) > 2:
            outputs_tensor = outputs_tensor.squeeze()

        #print("output_tensor_size")
        #print(outputs_tensor.size())

        # remove the input ids from output tensor
        # outputs_tensor = outputs_tensor[:, input_sequence_length:]

        outputs = outputs_tensor.tolist()
        for out_ids in outputs:
            eos_positions = [position for position, word_id in enumerate(out_ids) if word_id == eos_idx]
            if len(eos_positions) > 0:
                end_position = eos_positions[0]
                out_ids = out_ids[:end_position]
            out_text = tokenizer.decode(out_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            decode_out_sent_list = out_text.split("<n>")
            #decode_out_sent_list = nltk.tokenize.sent_tokenize(out_text)
            # output the predicted sentences to a file
            with open(join(args.pred_path, 'output/{}.dec'.format(num_exported_samples)), 'w') as f:
                f.write(make_html_safe('\n'.join(decode_out_sent_list)))
            num_exported_samples += 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path")
    parser.add_argument("--pred_path", default=None, type=str, required=True,
                        help="The path of output dir.")
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The path of the directory containing all the splits.")
    parser.add_argument("--split", default=None, type=str, required=True,
                        help="The split to be predicted.")
    # parser.add_argument("--prompt", type=str, default="")
    # parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--xlm_lang", type=str, default="", help="Optional language when used with the XLM model.")
    parser.add_argument("--max_output_length", type=int, default=120)
    parser.add_argument('--input_trunc_length', type=int, default=512,
                        help='Max length of output.')
    # parser.add_argument("--num_samples", type=int, default=1)
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
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    # parser.add_argument('--stop_token', type=str, default=None,
    #                    help="Token at which text generation is stopped")
    args = parser.parse_args()

    os.makedirs(args.pred_path)
    os.makedirs(join(args.pred_path, 'output'))

    json.dump(vars(args), open(join(args.pred_path, 'log.json'), 'w'))

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 1
    args.overwrite_cache = False
    # args.n_gpu = torch.cuda.device_count()

    set_seed(args)

    args.model_type = args.model_type.lower()

    model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path)
    model = model_class.from_pretrained(args.model_name_or_path)
    model.to(args.device)
    model.eval()

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
            args.repetition_penalty)


if __name__ == '__main__':
    main()
