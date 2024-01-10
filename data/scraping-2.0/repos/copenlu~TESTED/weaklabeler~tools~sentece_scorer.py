import torch
from torch import softmax

from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2TokenizerFast
from transformers import PreTrainedTokenizer
from typing import Union, List
from tqdm import tqdm

import numpy as np
 

def gpt_versioned_init(model_path: str = 'gpt2', cuda: bool = True) -> Union[torch.nn.Module, PreTrainedTokenizer]:

    if 'gpt2' in model_path.split("/")[-1][:4]:
        tokenizer = GPT2TokenizerFast.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)
    else:
        tokenizer = OpenAIGPTTokenizer.from_pretrained(model_path)
        model = OpenAIGPTLMHeadModel.from_pretrained(model_path)

    model.eval()
    if cuda:
        model.to('cuda')

    print('Sentece Assesment Model Has Initialised')

    return model, tokenizer


def sent_scoring(model_metas:Union[torch.nn.Module, PreTrainedTokenizer] = None , text:str = '', cuda: bool = True) -> float:
    sentence_prob = None
    try:
        model = model_metas[0]
        tokenizer = model_metas[1]
        if model is None or tokenizer is None:
            raise Exception('The model/tokenizer was not loaded properly')

        input_ids = torch.tensor(tokenizer(text, truncation=True)['input_ids']).unsqueeze(0)
        
        if cuda:
            input_ids = input_ids.to('cuda')

        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)

        loss, logits = outputs[:2]
        sentence_prob = loss.item()
            
    except RuntimeError as e:
        print("Unable to get the sentence probabilty with error: ",e)
        return sentence_prob
    return sentence_prob