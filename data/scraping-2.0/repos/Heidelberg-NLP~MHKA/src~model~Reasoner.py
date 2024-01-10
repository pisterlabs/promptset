# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
""" Finetuning the library models for multiple choice (Bert, Roberta, XLNet)."""

'''
Much of this code is taken from HuggingFace's repo:
https://github.com/huggingface/transformers/tree/master/examples
'''
from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import random
import numpy as np
import torch
import torch.nn as nn
import json
import logging
import math
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange

from transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForMultipleChoice, BertTokenizer,
                                  XLNetConfig, XLNetForMultipleChoice,
                                  XLNetTokenizer, RobertaConfig,
                                  RobertaForMultipleChoice, RobertaTokenizer, get_linear_schedule_with_warmup)

from transformers import AdamW
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForMaskedLM,
    BertTokenizer,
    CamembertConfig,
    CamembertForMaskedLM,
    CamembertTokenizer,
    DistilBertConfig,
    DistilBertForMaskedLM,
    DistilBertTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTConfig,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    PreTrainedTokenizer,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)
from text_utils import TextEncoder
from configuration_openai import OpenAIGPTConfig
from file_utils import add_start_docstrings
from modeling_utils import Conv1D, PreTrainedModel, SequenceSummary, prune_conv1d_layer
from utils_multiple_choice_know import (convert_examples_to_features, processors)
from sklearn.metrics import accuracy_score
logger = logging.getLogger(__name__)

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, RobertaConfig)), ())

MODEL_CLASSES = {
    "gpt2": (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    "openai-gpt": (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMultipleChoice, BertTokenizer),
    'xlnet': (XLNetConfig, XLNetForMultipleChoice, XLNetTokenizer),
    'roberta': (RobertaConfig, RobertaForMultipleChoice, RobertaTokenizer)
}

                                                  ###### Reasoning Cell Transformer ###### 
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)


ACT_FNS = {"relu": nn.ReLU, "swish": swish, "gelu": gelu}

OPENAI_GPT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "openai-gpt": "https://s3.amazonaws.com/models.huggingface.co/bert/openai-gpt-pytorch_model.bin"
}


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        """ Conv1D layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2)
            Basically works like a Linear layer but the weights are transposed
        """
        super().__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


def prune_conv1d_layer(layer, index, dim=1):
    """ Prune a Conv1D layer (a model parameters) to keep only entries in index.
        A Conv1D work as a Linear layer (see e.g. BERT) but the weights are transposed.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if dim == 0:
        b = layer.bias.clone().detach()
    else:
        b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = Conv1D(new_size[1], new_size[0]).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    new_layer.bias.requires_grad = False
    new_layer.bias.copy_(b.contiguous())
    new_layer.bias.requires_grad = True
    return new_layer

class Attention_Knowledge(nn.Module):
    def __init__(self, nx, n_ctx, n_ctx_know, cfg, scale=False):
        super(Attention_Knowledge, self).__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        assert n_state % cfg["n_head"] == 0
        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self.n_head = cfg["n_head"]
        self.split_size = n_state
        self.scale = scale
        self.output_attentions = cfg["output_attentions"]

        self.c_attn = Conv1D(n_state * 3, nx)
        self.c_proj = Conv1D(n_state, nx)
        self.attn_dropout = nn.Dropout(cfg["attn_pdrop"])
        self.resid_dropout = nn.Dropout(cfg["resid_pdrop"])
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.n_head, self.split_size // self.n_head)
        heads = set(heads) - self.pruned_heads
        for head in heads:
            head -= sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()
        index_attn = torch.cat([index, index + self.split_size, index + (2 * self.split_size)])
        # Prune conv1d layers
        self.c_attn = prune_conv1d_layer(self.c_attn, index_attn, dim=1)
        self.c_proj = prune_conv1d_layer(self.c_proj, index, dim=0)
        # Update hyper params
        self.split_size = (self.split_size // self.n_head) * (self.n_head - len(heads))
        self.n_head = self.n_head - len(heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def _attn(self, q, k, v, attention_mask=None, head_mask=None):

        w = torch.matmul(q, k)
        if self.scale:
            w = w / math.sqrt(v.size(-1))
        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
  						                             #Knowledge Attention Score 
        #f = open("result/attention.txt",'a')
        #f.write(str(w.tolist())+'\n') 
        #f.close()

        outputs = [torch.matmul(w, v)]
        if self.output_attentions:
            outputs.append(w)
        return outputs

    def merge_heads(self, x):
        x = x.permute(0, 2, 1, 3).contiguous()
        new_x_shape = x.size()[:-2] + (x.size(-2) * x.size(-1),)
        return x.view(*new_x_shape)  # in Tensorflow implem: fct merge_states

    def split_heads(self, x, k=False):
        new_x_shape = x.size()[:-1] + (self.n_head, x.size(-1) // self.n_head)
        x = x.view(*new_x_shape)  # in Tensorflow implem: fct split_states
        if k:
            return x.permute(0, 2, 3, 1)
        else:
            return x.permute(0, 2, 1, 3)

    def forward(self, x, xk, attention_mask=None, head_mask=None):

        xk = self.c_attn(xk)
        query, key, value = xk.split(self.split_size, dim=2)
        query = self.split_heads(x)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask)
        a = attn_outputs[0]
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a] + attn_outputs[1:]
        return outputs  # a, (attentions)


class MLP(nn.Module):
    def __init__(self, n_state, cfg):  # in MLP: n_state=3072 (4 * n_embd)
        super(MLP, self).__init__()
        nx = cfg["n_embd"]
        self.c_fc = Conv1D(n_state, nx)
        self.c_proj = Conv1D(nx, n_state)
        self.act = ACT_FNS[cfg["afn"]]
        self.dropout = nn.Dropout(cfg["resid_pdrop"])

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.c_proj(h)
        return self.dropout(h2)

class Block(nn.Module):
    def __init__(self, n_ctx, n_ctx_know, cfg, scale=False):
        super(Block, self).__init__()
        nx = cfg["n_embd"]
        self.attn_1 = Attention_Knowledge(nx, n_ctx, n_ctx_know, cfg, scale)
        self.ln_1 = nn.LayerNorm(nx, eps=cfg["layer_norm_epsilon"])
        self.mlp = MLP(4 * nx, cfg)
        self.ln_2 = nn.LayerNorm(nx, eps=cfg["layer_norm_epsilon"])
        #self.ln_3 = nn.LayerNorm(nx, eps=cfg["layer_norm_epsilon"]) 

    def forward(self, x, xk, attention_mask=None, head_mask=None):
        #xk = knowledge encoding
        #x = context encoding
        attn_outputs = self.attn_1(x, xk, attention_mask=attention_mask, head_mask=head_mask)
        k = attn_outputs[0]
        p = self.ln_1(x + k) ## knowledge_cross_attention
        m = self.mlp(p)
         
        h = self.ln_2(p + m)

        outputs = [h] + attn_outputs[1:]
        return outputs

class ReasonerModel(nn.Module):
    
    def __init__(self, cfg, vocab_size=50265, n_ctx=80, n_ctx_know=1024):
        super(ReasonerModel, self).__init__()

        self.output_attentions = True#cfg.output_attentions
        self.output_hidden_states = True #cfg.output_hidden_states
        n_positions = 80
        self.tokens_embed = nn.Embedding(vocab_size, cfg["n_embd"])
        self.positions_embed = nn.Embedding(n_positions, cfg["n_embd"])
        self.positions_knowledge_embed = nn.Embedding(n_positions, cfg["n_embd"])
        self.drop = nn.Dropout(cfg["embd_pdrop"])
        self.drop_1 = nn.Dropout(cfg["embd_pdrop"])
        self.h = nn.ModuleList([Block(n_ctx, n_ctx_know, cfg, scale=True) for _ in range(cfg["n_layer"])])
        #print(cfg["n_layer"])
        
        #self.init_weights()

    def get_input_embeddings(self):
        return self.tokens_embed

    def set_input_embeddings(self, new_embeddings):
        self.tokens_embed = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.h[layer].attn.prune_heads(heads)

    def forward(
        self,
        token_ids =None,
        input_ids=None,
        input_ids_know=None,
        position_knowledge_embeds=None, 
        knowledge_type_embeds=None, 
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = token_ids.size()
            #print(input_shape)
            #knowledge_shape = knowledge_ids.size()
            token_ids = token_ids.view(-1, input_shape[-1])
            #print(token_ids.size())
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")
        print(position_ids)
         
        if position_ids is None:
            # Code is different from when we had a single embedding matrice from position and token embeddings
            device = token_ids.device if token_ids is not None else inputs_embeds.device
            position_ids = torch.arange(input_shape[-1], dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])
        #position_knowledge_embeds.to(device)
       
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * 4  #cfg["n_layer"]
        if inputs_embeds is None:
            inputs_embeds = input_ids #self.tokens_embed(input_ids)
        position_embeds = self.positions_embed(position_ids)
        #position_knowledge_embeds = self.positions_knowledge_embed(position_knowledge_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.tokens_embed(token_type_ids)
        else:
            token_type_embeds = 0
        
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)  
        output_shape = input_shape + (hidden_states.size(-1),)
           
        all_attentions = ()
        all_hidden_states = ()
        for i, block in enumerate(self.h):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(hidden_states, input_ids_know, attention_mask, head_mask[i])
            hidden_states = outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

        outputs = (hidden_states.view(*output_shape),)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last hidden state, (all hidden states), (all attentions)


class DoubleReasonerModel(nn.Module):

    def __init__(self, cfg, clf_token, vocab_size=50265, n_ctx_seq=80, n_ctx=1024):
        super(DoubleReasonerModel, self).__init__()
        num_labels = 3
        self.transformer = ReasonerModel(cfg)
        self.lm_head = nn.Linear(cfg["n_embd"], vocab_size, bias=False)
        self.multiple_choice_head = SequenceSummary(cfg)

        #self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        token_ids =None,
        input_ids=None,
        input_ids_know=None,
        position_knowledge_embeds=None, 
        knowledge_type_embeds=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None,
    ):
        transformer_outputs = self.transformer(
            token_ids,
            input_ids,
            input_ids_know,
            position_knowledge_embeds, 
            knowledge_type_embeds,  
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,)
        hidden_states = transformer_outputs[0]
        hidden_states = hidden_states.view(-1, hidden_states.size(2), hidden_states.size(3))
        hidden_states = torch.mean(hidden_states, 1)
        return hidden_states
                                       
