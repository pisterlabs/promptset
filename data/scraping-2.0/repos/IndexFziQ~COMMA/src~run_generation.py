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
import numpy as np

from transformers import GPT2Config, OpenAIGPTConfig, XLNetConfig, TransfoXLConfig, XLMConfig, CTRLConfig

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer
from transformers import XLNetLMHeadModel, XLNetTokenizer
from transformers import TransfoXLLMHeadModel, TransfoXLTokenizer
from transformers import CTRLLMHeadModel, CTRLTokenizer
from transformers import XLMWithLMHeadModel, XLMTokenizer

# from tools.EA.get_concepts_from_behave import tokenization, lower_case
# from tools.EA.LIB.EVAL.bleu import compute_bleu
# from tools.EA.LIB.EVAL.rouge import compute_rouge_L
import sacrebleu
from rouge import Rouge

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
# PADDING_TEXT = """ In 1991, the remains of Russian Tsar Nicholas II and his family
# (except for Alexei and Maria) are discovered.
# The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
# remainder of the story. 1883 Western Siberia,
# a young Grigori Rasputin is asked by his father and a group of men to perform magic.
# Rasputin has a vision and denounces one of the men as a horse thief. Although his
# father initially slaps him for making such an accusation, Rasputin watches as the
# man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
# the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
# with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


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

    # all_loss = []
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
            next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)
            # print(next_token_logits.size())
            # tmp_eval_loss = outputs[0][:, :, -1][-1][-1]
            # print(outputs[0][:, :, -1][-1])
            # print (tmp_eval_loss.size ())
            # cur_loss = np.abs(tmp_eval_loss.item())
            # print(cur_loss)
            # all_loss.append(cur_loss)

            # repetition penalty from CTRL (https://arxiv.org/abs/1909.05858)
            for i in range(num_samples):
                for _ in set(generated[i].tolist()):
                    next_token_logits[i, _] /= repetition_penalty
                
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            if temperature == 0: # greedy sampling:
                next_token = torch.argmax(filtered_logits, dim=-1).unsqueeze(-1)
            else:
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)

            generated = torch.cat((generated, next_token), dim=1)
            # print(next_token.item())
            if next_token.item() == 29:
                break

    # print(all_loss)
    # avg_loss = np.mean (np.array (all_loss))
    # ppl = np.exp(avg_loss)
    # print('test_loss (bpe): %.4f, test_perplexity: %.4f' % (avg_loss, ppl))

    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default='gpt2', type=str, required=False,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default='/data1/data-xlx/checkpoints_ft_with_tapt/lmft_lm_gpt2_tapt_gn1_bp1_gc32_lr2_l100_e3_seed42_cgn1', type=str, required=False,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--source_text_path", type=str, default="/data1/data-xyq/ea_data/test.tsv")
    parser.add_argument("--prompt", type=str, default="I am happy")
    parser.add_argument("--padding_text", type=str, default="")
    parser.add_argument("--xlm_lang", type=str, default="", help="Optional language when used with the XLM model.")
    parser.add_argument("--length", type=int, default=60)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0,
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--repetition_penalty", type=float, default=1.2,
                        help="primarily useful for CTRL model; in that result, use 1.2")
    parser.add_argument("--top_k", type=int, default=1)
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
        if args.temperature > 0.7:
            logger.info('CTRL typically works better with lower temperatures (and lower top_k).')

    while True:
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

        # XLM masked-language modeling (MLM) models need masked token (see details in sample_sequence)
        is_xlm_mlm = args.model_type in ["xlm"] and 'mlm' in args.model_name_or_path
        if is_xlm_mlm:
            xlm_mask_token = tokenizer.mask_token_id
        else:
            xlm_mask_token = None

        raw_text = args.prompt if args.prompt else input("Model prompt >>> ")
        if args.model_type in ["transfo-xl", "xlnet"]:
            # Models with memory likes to have a long prompt for short inputs.
            raw_text = (args.padding_text if args.padding_text else PADDING_TEXT) + raw_text
        context_tokens = tokenizer.encode(raw_text, add_special_tokens=False)
        if args.model_type == "ctrl":
            if not any(context_tokens[0] == x for x in tokenizer.control_codes.values()):
                logger.info("WARNING! You are not starting your generation from a control code so you won't get good results")


        with open (args.source_text_path, 'r', encoding='utf-8') as data_file, \
                open (args.source_text_path + "_generated_tapt", 'w', encoding='utf-8') as new_file, \
                open (args.source_text_path + "_eval_tapt", 'w', encoding='utf-8') as eval_file:
            import csv
            reader = csv.reader (data_file, delimiter="\t")
            next (reader)

            all_data = []
            for line in reader:
                all_data.append (line)
            print (all_data[0])

            new_file.write ('source\ttarget\n')
            # generate batch for SRL
            # data_chunk = [all_data[i:i + args.batch] for i in range (0, len (all_data), args.batch)]
            # ROC Stories
            # index_of_sample = [2,3,4,5,6]
            # Story Commonsense
            # index_of_behavior = [3]
            # index_of_context = [2]
            from tqdm import tqdm
            # for data_sent in tqdm (data_chunk):
            source_corpus = []
            target_corpus = []
            for raw_data in tqdm(all_data):
                # import spacy
                # nlp = spacy.load ('en_core_web_sm')
                # doc = nlp (raw_data[3])  # only the current story line
                human = raw_data[1]
                source = raw_data[3]

                label_motive = raw_data[6]
                emotion_expect = raw_data[7]
                label_emotion = raw_data[8]

                emotions_set = 'default'
                if label_emotion == '1':
                    emotions_set = 'joy'
                elif label_emotion == '2':
                    emotions_set = 'trust'
                elif label_emotion == '3':
                    emotions_set = 'fear'
                elif label_emotion == '4':
                    emotions_set = 'surprise'
                elif label_emotion == '5':
                    emotions_set = 'sadness'
                elif label_emotion == '6':
                    emotions_set = 'disgust'
                elif label_emotion == '7':
                    emotions_set = 'anger'
                elif label_emotion == '8':
                    emotions_set = 'anticipation'

                human_needs = ['physiological', 'stability', 'love', 'esteem', 'spiritual growth']

                appendix = human_needs[int (label_motive) - 1]

                # concepts = tokenization (source)
                # concepts = lower_case (concepts)
                concepts = raw_data[5].split(',')
                # target = ''
                # if len (concepts) > 0:
                import random
                # text_ = "<Mot> The motive of %s is %s . </Mot>" % (human, appendix) +  "The emotion of %s could be %s ." % (human, emotions_set) + 'Situation: '+', '.join(concepts) + "<Beh>"
                # text_pair = event + "</Beh>"
                # target = "<mot> The motive of %s is %s . </mot> " % (human, appendix) + "<beh>"
                # target = "<emo> The emotion of %s could be %s . </emo> " % (human, emotions_set) + "<beh>"

                target = "<mot> The motive of %s is %s . </mot> " % (human, appendix) + "<emo> The emotion of %s could be %s . </emo> " % (human, emotions_set) + "<beh>"
                #          +  '<cpt> ' + ', '.join (random.choice (concepts)) + '</cpt> ' +  "<beh>"
                context_tokens = tokenizer.encode (target, add_special_tokens=False)
                    # text_pair = source + "</beh>"
                    # target = text_ + text_pair

                out = sample_sequence (
                    model=model,
                    context=context_tokens,
                    num_samples=args.num_samples,
                    length=args.length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    repetition_penalty=args.repetition_penalty,
                    is_xlnet=bool (args.model_type == "xlnet"),
                    is_xlm_mlm=is_xlm_mlm,
                    xlm_mask_token=xlm_mask_token,
                    xlm_lang=xlm_lang,
                    device=args.device,
                )
                out = out[:, len (context_tokens):].tolist ()

                for o in out:
                    # text = tokenizer.decode(o, cleanI_up_tokenization_spaces=True)
                    text = tokenizer.decode (o)
                    text = text[: text.find (args.stop_token) if args.stop_token else None]

                    target = text.replace ('</beh>', '')
                source_corpus.append(source)
                target_corpus.append(target)

                new_file.write (source+"\t"+target+"\n")

            rouge_ = []
            bleu_list = []

            for source, target in zip (source_corpus, target_corpus):

                bleu = sacrebleu.corpus_bleu(target, source)
                # print(type(bleu.precisions))
                # bleu_1 = bleu.precisions
                bleu_list.append(list(bleu.precisions))
            # bleu = compute_bleu ([[j.split ()] for j in source_corpus], [i.split () for i in target_corpus])
            # from rouge import rouge_score

            rouge = Rouge ()

            for source, target in zip(source_corpus, target_corpus):
                # rouge_l = compute_rouge_L(source, target)
                rouge_x = rouge.get_scores(source, target)
                rouge_1 = rouge_x[0]["rouge-1"]['f']
                rouge_2 = rouge_x[0]["rouge-2"]['f']
                rouge_l = rouge_x[0]["rouge-l"]['f']
                rouge_.append([rouge_1,rouge_2,rouge_l])

            bleu_score = np.mean(bleu_list,axis=0)
            rouge_score_ = np.mean(rouge_,axis=0)

            print ("BLEU-1 score: " + str (bleu_score[0]) + "\n")
            print ("BLEU-2 score: " + str (bleu_score[1]) + "\n")
            print ("BLEU-4 score: " + str (bleu_score[3]) + "\n")
            print ("Rouge-1 score: " + str (rouge_score_[0] * 100) + "\n")
            print ("Rouge-2 score: " + str (rouge_score_[1] * 100) + "\n")
            print ("Rouge-L score: " + str (rouge_score_[2] * 100) + "\n")

            eval_file.write("BLEU-1 score: " + str (bleu_score[0]) +"\n")
            eval_file.write("BLEU-2 score: " + str (bleu_score[1]) + "\n")
            eval_file.write("BLEU-4 score: " + str (bleu_score[3]) + "\n")
            eval_file.write("Rouge-1 score: " + str (rouge_score_[0]*100) + "\n")
            eval_file.write("Rouge-2 score: " + str (rouge_score_[1]*100) + "\n")
            eval_file.write("Rouge-L score: " + str (rouge_score_[2]*100) + "\n")


        if args.prompt:
            break

    # return target


if __name__ == '__main__':
    main()
