# # Copyright (c) 2019-present, HuggingFace Inc.
# All rights reserved.
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import logging
import random
from argparse import ArgumentParser
from itertools import chain
from pprint import pformat
import warnings
from collections import defaultdict


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import numpy as np

from pytorch_transformers import OpenAIGPTLMHeadModel, OpenAIGPTTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from train import build_input_from_segments, add_special_tokens_, pad_dataset, get_data_loaders
from utils import get_dataset, download_pretrained_model

from functools import partial
from entmax import SparsemaxLoss, Entmax15Loss, EntmaxBisectLoss, sparsemax, entmax15, entmax_bisect

import eval_utils
from scipy.stats import entropy

SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
MODEL_INPUTS = ["input_ids", "reply", "token_type_ids"]


def build_input(persona, history, tokenizer, lm_labels=False, with_eos=True, speaker=None):
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    sequence = [[bos] + list(chain(*persona))] + [history]
    if speaker==1:
        sequence = [sequence[0]] + [[speaker1 if (len(sequence)-i) % 2 else speaker2] + s for i, s in enumerate(sequence[1:])] + [[speaker1]]
    else:
        sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])] + [[speaker2]]
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    if speaker==1:
        instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    else:
        instance["token_type_ids"] = [speaker1 if i == 0  else speaker1 if i % 2 else speaker2 for i, s in enumerate(sequence) for _ in s]
    return instance

def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
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

def softmax_temperature(X, temperature = 1.0, axis = None):
    
    X = X*(1/temperature)
    p = torch.softmax(X, dim=-1)
    return p

def run():
    parser = ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="", help="Path or url of the dataset. If empty download from S3.")
    parser.add_argument("--dataset_cache", type=str, default='./dataset_cache', help="Path or url of the dataset cache")
    parser.add_argument("--model", type=str, default="gpt2", help="Model type (openai-gpt or gpt2)", choices=['openai-gpt', 'gpt2', 'gpt2-medium'])  # anything besides gpt2 will load openai-gpt
    parser.add_argument("--model_checkpoint", type=str, default="", help="Path, url or short name of the model")
    parser.add_argument("--max_history", type=int, default=2, help="Number of previous utterances to keep in history")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--batch_size", type=int, default=1)    
    parser.add_argument("--no_sample", action='store_true', help="Set to use greedy decoding instead of sampling")
    parser.add_argument("--max_length", type=int, default=20, help="Maximum length of the output utterances")
    parser.add_argument("--min_length", type=int, default=1, help="Minimum length of the output utterances")
    parser.add_argument("--num_candidates", type=int, default=1)
    parser.add_argument("--personality_permutations", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0, help="Seed")
    parser.add_argument("--distributed", action='store_true') 
    parser.add_argument("--temperature", type=int, default=1, help="Sampling softmax temperature")
    parser.add_argument("--top_k", type=int, default=0, help="Filter top-k tokens before sampling (<=0: no filtering)")
    parser.add_argument("--top_p", type=float, default=0, help="Nucleus filtering (top-p) before sampling (<=0.0: no filtering)")
    parser.add_argument("--entmax_alpha", type=float, default=1.5)
    parser.add_argument("--entmax_k", type=int, default=512)
    parser.add_argument("--entmax_bisect_iter", type=int, default=50)
    parser.add_argument("--loss", default="cross_entropy", type=str)
    parser.add_argument("--metric", default="bleu", type=str)
    parser.add_argument("--epsilon", default=0.000001, type=float)
    parser.add_argument("--name", default='', type=str)
    parser.add_argument("--temp", default=0, type=float)


    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__file__)
    logger.info(pformat(args))

    args.train_batch_size=args.batch_size
    args.valid_batch_size=args.batch_size

    generic_entmax_loss = partial(
        EntmaxBisectLoss,
        alpha=args.entmax_alpha,
        n_iter=args.entmax_bisect_iter)

    
    loss_funcs = {"cross_entropy": nn.CrossEntropyLoss,
                  "sparsemax": partial(SparsemaxLoss, k=args.entmax_k),
                  "entmax15": partial(Entmax15Loss, k=args.entmax_k),
                  "entmax": generic_entmax_loss,
                  "entmax_alpha": "entmax_alpha"}

    assert args.loss in loss_funcs
    loss_func = loss_funcs[args.loss]


    generic_entmax = partial(
        entmax_bisect,
        alpha=args.entmax_alpha,
        n_iter=args.entmax_bisect_iter)

    gen_funcs = {"softmax": torch.softmax,
                 "sparsemax": partial(sparsemax, k=args.entmax_k),
                 "entmax15": partial(entmax15, k=args.entmax_k),
                 "entmax": generic_entmax,
                 "entmax_alpha": "entmax_alpha"}

    if args.loss=="cross_entropy":
        gen_func = gen_funcs["softmax"]
    elif args.loss=="sparsemax":
        gen_func = gen_funcs["sparsemax"]
    elif args.loss=="entmax15":
        gen_func = gen_funcs["entmax15"]
    elif args.loss=="entmax":
        gen_func = gen_funcs["entmax"]
    elif args.loss=="entmax_alpha":
        gen_func = gen_funcs["entmax_alpha"]

    if args.model_checkpoint == "":
        if args.model == 'gpt2' or args.model == 'gpt2-medium':
            raise ValueError("Interacting with GPT2 requires passing a finetuned model_checkpoint")
        else:
            args.model_checkpoint = download_pretrained_model()
    
    
    if args.seed != 0:
        random.seed(args.seed)
        torch.random.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)


    logger.info("Get pretrained model and tokenizer")
    tokenizer_class, model_class = (GPT2Tokenizer, GPT2LMHeadModel) if args.model == 'gpt2' or args.model == 'gpt2-medium' else (OpenAIGPTTokenizer, OpenAIGPTLMHeadModel)
    tokenizer = tokenizer_class.from_pretrained(args.model_checkpoint)
    model = model_class.from_pretrained(args.model_checkpoint)
    model.to(args.device)
    add_special_tokens_(model, tokenizer)

    personachat = get_dataset(tokenizer, args.dataset_path, args.dataset_cache)

    bos, eos, speaker1, speaker2, pad = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    model.eval()
    datasets = {"train": defaultdict(list), "valid": defaultdict(list)}

    personalities = [dialog["personality"] for dataset in personachat.values() for dialog in dataset]
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)

    lengths=[]
    
    f = open('./conv_sim/model_to_model_' + args.name, 'w')
    for dataset_name, dataset in personachat.items():
        num_candidates = 1
        if dataset_name != 'train':
            for dialog in dataset:
                persona_1 = random.choice(personalities)
                persona_2 = random.choice(personalities)
                utterance = dialog["utterances"][0]
                history = utterance["history"][0]
                instance_1 = build_input(persona_1, history, tokenizer, speaker=1)
                instance_2 = build_input(persona_2, history, tokenizer, speaker=2)
                
                v=0
                input_1 = instance_1["input_ids"]
                input_2 = instance_2["input_ids"]
                token_ids_1 = instance_1["token_type_ids"]
                token_ids_2 = instance_2["token_type_ids"]
                
                conversation=[]
                while True:
                    if v%2==0 or v==0:
                        inpu = torch.tensor(input_1).unsqueeze(0).cuda()
                        token_ids = torch.tensor(token_ids_1).unsqueeze(0).cuda()
                    else:
                        inpu = torch.tensor(input_2).unsqueeze(0).cuda()
                        token_ids = torch.tensor(token_ids_2).unsqueeze(0).cuda()
                    current_output=[]
                    for i in range(args.max_length):
                        if i > 0:
                            inpu = torch.cat([inpu , prev.unsqueeze(0)],1)  
                            if token_ids[0][-1]==50260:
                                token_ids = torch.cat([token_ids, torch.tensor([50260]).cuda().unsqueeze(0)],1)
                                if v%3==0 or v==1:
                                    token_ids_1.append(50260)
                                    token_ids_2.append(50261)
                                else:
                                    token_ids_1.append(50261)
                                    token_ids_2.append(50260)
                            else:
                                token_ids = torch.cat([token_ids, torch.tensor([50261]).cuda().unsqueeze(0)],1)   
                                token_ids_1.append(50261)
                                token_ids_2.append(50260)
                                if v%3==0 or v==1:
                                    token_ids_1.append(50261)
                                    token_ids_2.append(50260)
                                else:
                                    token_ids_1.append(50260)
                                    token_ids_2.append(50261)
                        if token_ids.size(1)!=inpu.size(1):
                            break
                        logits = model(inpu, token_type_ids=token_ids)
                        if isinstance(logits, tuple):  
                            logits = logits[0]
                        logits = logits[0, -1, :] 
                        if args.top_k>0 or args.top_p >0:
                            logits = top_filtering(logits, top_k=args.top_k, top_p=args.top_p)
                        if args.temp==0:
                            probs = gen_func(logits, dim=-1)
                        else:
                            probs=softmax_temperature(logits,temperature=args.temp, axis=1)

                        prev = torch.multinomial(probs, 1)
                        if prev.item() in special_tokens_ids:
                            break
                        current_output.append(prev.item())
                    input_1.extend(current_output)
                    
                    input_2.extend(current_output)
                    if v%2==0 or v==0:
                        input_1.append(50261)
                        input_2.append(50260)
                        token_ids_1.append(50261)
                        token_ids_2.append(50260)
                    else:
                        input_1.append(50260)
                        input_2.append(50261)
                        token_ids_1.append(50260)
                        token_ids_2.append(50261)
                    v+=1
                    conversation.append(current_output)
                    if v==20 or len(current_output)==0:
                        print('empty')
                        break
                    c=0
                    if len(conversation)>2:
                        for word in current_output:
                            if word in conversation[-3]:
                                c+=1

                        if c/len(current_output)>=0.8:
                            break 
                    c=0
                    if len(conversation)>1:
                        for word in current_output:
                            if word in conversation[-2]:
                                c+=1
                        
                        if c/len(current_output)>=0.8:
                            break 


                print('persona_1: ', tokenizer.decode(chain(*persona_1)))
                print('persona_2: ', tokenizer.decode(chain(*persona_2)))
                print('history: ', tokenizer.decode(history))
                for utt in conversation:
                    print('-----', tokenizer.decode(utt))
                print('\n')
                
                f.write('persona_1: '+ str(tokenizer.decode(chain(*persona_1))))
                f.write('persona_2: '+ str(tokenizer.decode(chain(*persona_2))))
                f.write('history: ' + str(tokenizer.decode(history)))
                for utt in conversation:
                    f.write(tokenizer.decode(utt, clean_up_tokenization_spaces=False))
                    f.write('\n')


                lengths.append(len(conversation))
    print(len(lengths))
    print('average number of turns:', np.array(lengths).mean())

            
            



if __name__ == "__main__":
    run()
