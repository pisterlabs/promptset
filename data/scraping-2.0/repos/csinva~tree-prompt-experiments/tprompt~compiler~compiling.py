# from evaluator import PromptHooker, modify_activations
import imodelsx.treeprompt.stump
from sklearn.preprocessing import OneHotEncoder
import sklearn.tree
import random
import joblib
from dict_hash import sha256
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import imodelsx.process_results
from collections import defaultdict
import numpy as np
from copy import deepcopy
import transformers
import sys
import tprompt.utils
from os.path import join
import datasets
from typing import Dict, List
from sklearn.tree import plot_tree
import imodelsx.util
import imodelsx.metrics
import numpy as np
import tprompt.utils
from scipy.special import softmax
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import torch.cuda
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.base import BaseEstimator, ClassifierMixin
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch
import torch
import math
import vec2text
import openai
openai.api_key = open(os.path.expanduser('~/.openai_api_key')).read().strip()

OUTPUTS_ALL = {}
PROMPT_NUM_GLOBAL = 0


def get_avg_soft_prompt(checkpoint, prompts):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForCausalLM.from_pretrained(checkpoint).eval()

    def store_activations(module, inputs, outputs):
        global OUTPUTS_ALL
        global PROMPT_NUM_GLOBAL
        OUTPUTS_ALL[PROMPT_NUM_GLOBAL] = outputs.detach().cpu()
        PROMPT_NUM_GLOBAL += 1
        return outputs

    hook = model.transformer.drop.register_forward_hook(store_activations)
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt")
        # hook = model.transformer.h[3].register_forward_hook(change_activations)
        _ = model(**inputs)

    hook.remove()
    assert len(OUTPUTS_ALL) == len(prompts)

    # most_probable_tokens = torch.topk(logits_modified, k=10, dim=-1)
    # print('\n'.join([tokenizer.decode(x)
    #   for x in most_probable_tokens.indices[0, -1]]))
    # logits_orig = model(**inputs).logits

    vals = list(OUTPUTS_ALL.values())
    emb_size = vals[0].shape[-1]

    max_len = max([x.shape[1] for x in vals])
    # add left padding
    padded = [torch.cat([torch.zeros((1, max_len - x.shape[1], emb_size)), x], dim=1)
              for x in vals]

    # average
    avg = torch.concat(tuple(padded)).mean(axis=0).unsqueeze(0)
    return avg


def get_avg_inverted_text_prompt(prompts: List[str]) -> str:
    def _get_embeddings_openai(text_list, model="text-embedding-ada-002", cache_dir=os.path.expanduser('~/.openai_emb_cache')) -> torch.Tensor:
        batches = math.ceil(len(text_list) / 128)
        outputs = []
        for batch in range(batches):
            text_list_batch = text_list[batch * 128: (batch + 1) * 128]

            # check for cache
            cache_path = join(cache_dir, sha256({'embs': text_list_batch}))
            if os.path.exists(cache_path):
                outputs.extend(joblib.load(cache_path))
            else:
                response = openai.Embedding.create(
                    input=text_list_batch,
                    model=model,
                    # override default base64 encoding...
                    encoding_format="float",
                )
                embs = [e["embedding"] for e in response["data"]]
                outputs.extend(embs)

                # save to cache
                if cache_dir is not None:
                    if not os.path.exists(cache_dir):
                        os.makedirs(cache_dir)
                    joblib.dump(embs, cache_path)
        return torch.tensor(outputs)

    embeddings = _get_embeddings_openai(prompts)
    avg_embedding = embeddings.mean(dim=0, keepdim=True).cuda()
    corrector = vec2text.load_corrector("text-embedding-ada-002")
    avg_text = vec2text.invert_embeddings(
        embeddings=avg_embedding,
        corrector=corrector
    )
    return avg_text
