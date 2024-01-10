import os
import math
import logging
from pprint import pformat
from argparse import ArgumentParser
from collections import defaultdict
from itertools import chain

import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, TensorDataset
from pytorch_pretrained_bert import OpenAIGPTTokenizer
from pytorch_pretrained_bert import OpenAIGPTLMHeadModel
#from pytorch_pretrained_bert import OpenAIGPTDoubleHeadsModel
from optim import AdamW

NO_SAMPLE = False
MAX_HISTORY = 2
MAX_LENGTH = 20
MIN_LENGTH = 1
temperature = .7
DEVICE = 0
TOP_K= 0
TOP_P= .9
WEIGHT_PATH = '/media/sec/conv_ai_weights/3.pth'
DATA_PATH = 'raw_dataset.pyobj'


SPECIAL_TOKENS = ["<bos>", "<eos>", "<speaker1>", "<speaker2>", "<pad>"]
ATTR_TO_SPECIAL_TOKEN = {'bos_token': '<bos>', 'eos_token': '<eos>', 'pad_token': '<pad>',
                         'additional_special_tokens': ['<speaker1>', '<speaker2>']}
MODEL_INPUTS = ["input_ids", "mc_token_ids", "lm_labels", "mc_labels", "token_type_ids"]
PADDED_INPUTS = ["input_ids", "lm_labels", "token_type_ids"]

def add_special_tokens_(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    num_added_tokens = tokenizer.set_special_tokens(SPECIAL_TOKENS) # doesn't add if they are already there
    model.set_num_special_tokens(len(SPECIAL_TOKENS))
    #orig_num_tokens = len(tokenizer.encoder)
    #num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there 
    #if num_added_tokens > 0:
        #model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)

tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
model = OpenAIGPTLMHeadModel.from_pretrained('openai-gpt')
#model = OpenAIGPTDoubleHeadsModel.from_pretrained('openai-gpt')

add_special_tokens_(model, tokenizer)
weight = torch.load(WEIGHT_PATH)
model.load_state_dict( weight, strict= False)
model.cuda(0)
model.eval()

dataset = torch.load('raw_dataset.pyobj')
P = [dialog['personality'] for dialog in dataset['valid']]


def build_input_from_segments(persona, history, reply, tokenizer, lm_labels=False, with_eos=True):
    """ Build a sequence of input from 3 segments: persona, history and last reply. """
    bos, eos, speaker1, speaker2 = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1])
    sequence = [[bos] + list(chain(*persona))] + history + [reply + ([eos] if with_eos else [])]
    sequence = [sequence[0]] + [[speaker2 if (len(sequence)-i) % 2 else speaker1] + s for i, s in enumerate(sequence[1:])]
    
    instance = {}
    instance["input_ids"] = list(chain(*sequence))
    instance["token_type_ids"] = [speaker2 if i % 2 else speaker1 for i, s in enumerate(sequence) for _ in s]
    instance["mc_token_ids"] = len(instance["input_ids"]) - 1
    
    if lm_labels:
        instance["lm_labels"] = ([-1] * sum(len(s) for s in sequence[:-1])) + [-1] + sequence[-1][1:]
    else:
        instance["lm_labels"] = [-1] * len(instance["input_ids"])
        
    return instance


def top_filtering(logits, top_k=0., top_p=0.9, threshold=-float('Inf'), filter_value=-float('Inf')):
    assert logits.dim() == 1  # Only work for batch size 1 for now - could update but it would obfuscate    \ a bit the code
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

def sample_sequence(personality, history, tokenizer, model, current_output=None):                 
    special_tokens_ids = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS)
    if current_output is None:
        current_output = []
                                                                                                        
    for i in range(MAX_LENGTH):                                                                    
        instance = build_input_from_segments(personality, history, current_output, tokenizer, with_eos=False)

        input_ids = torch.tensor(instance["input_ids"], device=DEVICE).unsqueeze(0)                
        token_type_ids = torch.tensor(instance["token_type_ids"], device=DEVICE).unsqueeze(0)
        logits = model(input_ids, token_type_ids=token_type_ids)
        if isinstance(logits, tuple):  # for gpt2 and maybe others
            logits = logits[0]
        logits = logits[0, -1, :] / temperature                                                    
        logits = top_filtering(logits, top_k=TOP_K, top_p=TOP_P)                              
        probs = F.softmax(logits, dim=-1)

        prev = torch.topk(probs, 1)[1] if NO_SAMPLE else torch.multinomial(probs, 1)               
        if i < MIN_LENGTH and prev.item() in special_tokens_ids:                                   
            while prev.item() in special_tokens_ids:
                if probs.max().item() == 1:
                    #warnings.warn("Warning: model generating special token with probability 1.")
                    break  # avoid infinitely looping over special token
                prev = torch.multinomial(probs, num_samples=1)

        if prev.item() in special_tokens_ids:
            break
        current_output.append(prev.item())

    return current_output 

import random
import torch.nn.functional as F 

p = random.choice(P)
print( tokenizer.decode( chain(*p) ) )

history = []
while True:
    raw_text = input('>>> ')
    while not raw_text:                                                                      
        print('Prompt should not be empty!')                                                        
        raw_text = input(">>> ")                                                                    

    if raw_text == 'reset':
        p = random.choice(P)
        history = []
        print('Reloading New Personality')
        print( tokenizer.decode( chain(*p) ) )
        continue

    history.append(tokenizer.encode(raw_text))                                                      
    with torch.no_grad():                                                                           
        out_ids = sample_sequence(p, history, tokenizer, model)                     
    history.append(out_ids)                                                                         
    history = history[-(2*MAX_HISTORY+1):]                                                     
    out_text = tokenizer.decode(out_ids, skip_special_tokens=True)                                  
    print(out_text)   

