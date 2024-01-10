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
from Reasoner import DoubleReasonerModel
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter
#from parallel import DataParallelModel, DataParallelCriterion
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

                                                  ###### Knowledge_encoding Transformer Block
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def swish(x):
    return x * torch.sigmoid(x)
#torch.manual_seed(0)
#torch.backends.cudnn.benchmark = False

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

class Attention(nn.Module):
    def __init__(self, nx, n_ctx, cfg, scale=False):
        super().__init__()
        n_state = nx  # in Attention: n_state=768 (nx=n_embd)
        # [switch nx => n_state from Block to Attention to keep identical to TF implem]
        #assert n_state % cfg["n_head"] == 0
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
        # w = w * self.bias + -1e9 * (1 - self.bias)  # TF implem method: mask_attn_weights
        # XD: self.b may be larger than w, so we need to crop it
        b = self.bias[:, :, : w.size(-2), : w.size(-1)]
        w = w * b + -1e4 * (1 - b)

        if attention_mask is not None:
            # Apply the attention mask
            #print("size")
            #print(attention_mask.size())
            #print(w.size())
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

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

    def forward(self, x, attention_mask=None, head_mask=None):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)

        attn_outputs = self._attn(query, key, value, attention_mask, head_mask)
        a = attn_outputs[0]
        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a] + attn_outputs[1:]
        #print(a.size())
        #print(query.size(), key.size(), value.size())
        return outputs  # a, (attentions)


class MLP(nn.Module):
    def __init__(self, n_state, cfg):  # in MLP: n_state=3072 (4 * n_embd)
        super().__init__()
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
    def __init__(self, n_ctx, cfg, scale=False):
        super().__init__()
        nx = cfg["n_embd"]
        self.attn = Attention(nx, n_ctx, cfg, scale)
        self.ln_1 = nn.LayerNorm(nx, eps=cfg["layer_norm_epsilon"])
        self.mlp = MLP(4 * nx, cfg)
        self.ln_2 = nn.LayerNorm(nx, eps=cfg["layer_norm_epsilon"])

    def forward(self, x, attention_mask=None, head_mask=None):
        attn_outputs = self.attn(x, attention_mask=attention_mask, head_mask=head_mask)
        a = attn_outputs[0]
        n = self.ln_1(x + a)
        m = self.mlp(n)
        h = self.ln_2(n + m)
        #print(h.size())
        #print(n.size())
        outputs = [h] + attn_outputs[1:]
        return outputs

class OpenAIGPTModel(nn.Module):
    
    def __init__(self, cfg, vocab_size=50265, n_ctx=1024):
        super(OpenAIGPTModel, self).__init__()

        self.output_attentions = True #cfg.output_attentions
        self.output_hidden_states = True #cfg.output_hidden_states
        n_positions = 1024 
        self.tokens_embed = nn.Embedding(vocab_size, cfg["n_embd"])
        self.positions_embed = nn.Embedding(n_positions, cfg["n_embd"])
        self.drop = nn.Dropout(cfg["embd_pdrop"])
        self.h = nn.ModuleList([Block(n_ctx, cfg, scale=True) for _ in range(cfg["n_layer"])])

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
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
    ):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if position_ids is None:
            # Code is different from when we had a single embedding matrice from position and token embeddings
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(input_shape[-1], dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, input_shape[-1])
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0
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
            inputs_embeds = self.tokens_embed(input_ids)
        position_embeds = self.positions_embed(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.tokens_embed(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)
        #print(hidden_states.size())
        all_attentions = ()
        all_hidden_states = ()
        for i, block in enumerate(self.h):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(hidden_states, attention_mask, head_mask[i])
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
         
        return outputs , position_embeds, token_type_embeds # last hidden state, (all hidden states), (all attentions)


class OpenAIGPTDoubleHeadsModel(nn.Module):

    def __init__(self, cfg, clf_token, vocab_size=50265, n_ctx=1024):
        super(OpenAIGPTDoubleHeadsModel, self).__init__()
        num_labels = 2
        self.transformer =  OpenAIGPTModel(cfg)
        self.lm_head = nn.Linear(cfg["n_embd"], vocab_size, bias=True)
        self.multiple_choice_head = SequenceSummary(cfg)
        self.expand = nn.Linear(cfg["n_embd"], 1024, bias=True)  
        #self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        mc_token_ids=None,
        lm_labels=None,
        mc_labels=None,
    ):
        transformer_outputs, position_embeds, token_type_embeds = self.transformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        hidden_states = transformer_outputs[0]
        print(hidden_states.size())
        hidden_states = hidden_states.view(-1, hidden_states.size(2), hidden_states.size(3))
        #hidden_states = self.expand(hidden_states)
        print(hidden_states.size())
        return hidden_states#, position_embeds, token_type_embeds 
        ######

def select_field(features, field):
    return [
        [
            choice[field]
            for choice in feature.choices_features
        ]
        for feature in features
    ]
   
def simple_accuracy(preds, labels):
    #print(preds,labels)
    accuracy = accuracy_score(labels, preds)

    return accuracy 

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

class Classifier(nn.Module):
    """docstring for Reasoning"nn.Module """
    def __init__(self, cfg):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(cfg["n_embd"], 1, bias=False)

    def forward(
        self,
        pooled_output=None,
    ):    
       logits = self.classifier(pooled_output)        
       return logits 


def train(args, train_dataset, model, tokenizer, encode_model, reasoner_model, linear_classifier):
    """ Train the model """
    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter()

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    scheduler = get_linear_schedule_with_warmup(
                    optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
                        )
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    if args.n_gpu > 1:
        encode_model = torch.nn.DataParallel(encode_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        encode_model = torch.nn.parallel.DistributedDataParallel(encode_model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    if args.n_gpu > 1:
        reasoner_model = torch.nn.DataParallel(reasoner_model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        reasoner_model = torch.nn.parallel.DistributedDataParallel(reasoner_model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)
    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    
    tr_loss, logging_loss = 0.0, 0.0
    best_dev_acc, best_dev_loss = 0.0, 99999999999.0
    best_steps = 0
    model.zero_grad()
    global_step = 0
    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            encode_model.train()
            reasoner_model.train()
            batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch[0],
                      'attention_mask': batch[1],
                      'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                      'labels':         batch[5]}

            outputs, pooled_output, attention_layer = model(**inputs) #Encoding Context 
            knowledge_inputs = {'input_ids': batch[3], 'attention_mask': batch[4]} 
            knowledge_output = encode_model(**knowledge_inputs) # Encoding Knowledge 
            
            

            enhanced_output = reasoner_model(batch[0], attention_layer, knowledge_output, None, None) # Reasoning Cell
            
            
            #pooled_output = pooled_output.view(-1, enhanced_output.size(1)) + enhanced_output.view(-1, enhanced_output.size(1))
            pooled_output = enhanced_output.view(-1, enhanced_output.size(1))


            logits = linear_classifier(pooled_output)
            logits = logits.view(-1, 2) 
            loss_fct = nn.CrossEntropyLoss()#.nn.nn.KLDivLoss()
            labels = batch[5]
            loss = loss_fct(logits, labels)

            if args.n_gpu > 1:
                loss = loss.mean() # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            if (step + 1) % args.gradient_accumulation_steps == 0:

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Log metrics
                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, encode_model, reasoner_model, linear_classifier, tokenizer, global_step)
                        
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                        if results["eval_acc"] > best_dev_acc:
                            best_dev_acc = results["eval_acc"]
                            best_dev_loss = results["eval_loss"]
                            best_steps = global_step
                            if args.do_test:
                                results_test = evaluate(args, model, encode_model, reasoner_model, linear_classifier, tokenizer, global_step, test=True)
                                for key, value in results_test.items():
                                    tb_writer.add_scalar('test_{}'.format(key), value, global_step)
                                logger.info("test acc: %s, loss: %s, global steps: %s", str(results_test['eval_acc']), str(results_test['eval_loss']), str(global_step))
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss)/args.logging_steps, global_step)
                    logger.info("Average loss: %s at global step: %s", str((tr_loss - logging_loss)/args.logging_steps), str(global_step))
                    logging_loss = tr_loss

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_vocabulary(output_dir)
                    PATH = args.output_dir+"/model.pt"
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    torch.save({'encode_state_dict': encode_model.state_dict(),
                                'reasoner_state_dict': reasoner_model.state_dict()}, PATH)

                    logger.info("Saving model checkpoint to %s", output_dir)

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step, best_steps

def evaluate(args, model, encode_model, reasoner_model, linear_classifier, tokenizer, global_step, prefix="", test=False):

    eval_task_names = (args.task_name,)
    eval_outputs_dirs = (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=not test, test=test)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation {} *****".format(prefix))
        logger.info("  Num examples = %d", len(eval_dataset))
        logger.info("  Batch size = %d", args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None
        exampleid = None
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            encode_model.eval()
            reasoner_model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],
                          'token_type_ids': batch[2] if args.model_type in ['bert', 'xlnet'] else None,  # XLM don't use segment_ids
                          'labels':         batch[5]}
                
                outputs, pooled_output, attention_layer = model(**inputs)
              
                knowledge_inputs = {'input_ids': batch[3], 'attention_mask':batch[4]}
                knowledge_output = encode_model(**knowledge_inputs)
                
                enhanced_output = reasoner_model(batch[0], attention_layer, knowledge_output, None, None)

                    
                #enhanced_output = reasoner_model(batch[0], attention_layer, knowledge_output, None, None)

                #pooled_output = pooled_output.view(-1, enhanced_output.size(1)) + enhanced_output.view(-1, enhanced_output.size(1))
                pooled_output = enhanced_output.view(-1, enhanced_output.size(1))
                
                logits = linear_classifier(pooled_output)
                #loss = outputs[0]  # model outputs are always tuple in transformers (see doc)
                logits = logits.view(-1, 2)
                #print(logits)
                loss_fct = nn.CrossEntropyLoss()
                labels = batch[5]
                ids = batch[6]       
                loss = loss_fct(logits, labels)
                tmp_eval_loss = loss
                eval_loss += tmp_eval_loss.mean().item()

            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                prediction = np.argmax(preds, axis=1)
                out_label_ids = inputs['labels'].detach().cpu().numpy()
                
                							#"error analysis"
                #if np.argmax(logits.detach().cpu().numpy(),-1)[0] != inputs['labels'].detach().cpu().numpy()[0]:
                #    print(ids.tolist()[0]) #, np.argmax(logits.detach().cpu().numpy(),-1)[0], inputs['labels'].detach().cpu().numpy())
                #if test==False: 
                #    for i in range(len(prediction)):
                #         print("Results:")
                #         print(ids.tolist()[i], out_label_ids[i], prediction[i])             
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                prediction = np.argmax(preds, axis=1)
                out_label_ids = np.append(out_label_ids, inputs['labels'].detach().cpu().numpy(), axis=0)
                #exampleid = np.append(exampleid, batch[5].detach().cpu().numpy(), axis=0)
                #if np.argmax(logits.detach().cpu().numpy(),-1)[0] != inputs['labels'].detach().cpu().numpy()[0]:
                #    print(ids.tolist()[0]) #np.argmax(logits.detach().cpu().numpy(), -1)[0], inputs['labels'].detach().cpu().numpy()[0])
                #if test==True: 
                #    for i in range(len(prediction)):
                #         print("Results:")
                #         print(ids.tolist()[i], out_label_ids[i], prediction[i]) #index(max(logits[i])))        
        eval_loss = eval_loss / nb_eval_steps
        preds = np.argmax(preds, axis=1)
        acc = simple_accuracy(preds, out_label_ids)

        result = {"eval_acc": acc, "eval_loss": eval_loss}
        results.update(result)
        
        output_eval_file = os.path.join(eval_output_dir, "is_test_" + str(test).lower() + str(global_step) +"_eval_results.txt")

        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(str(prefix) + " is test:" + str(test)))
            writer.write("model           =%s\n" % str(args.model_name_or_path))
            writer.write("total batch size=%d\n" % (args.per_gpu_train_batch_size * args.gradient_accumulation_steps *
                         (torch.distributed.get_world_size() if args.local_rank != -1 else 1)))
            writer.write("train num epochs=%d\n" % args.num_train_epochs)
            writer.write("fp16            =%s\n" % args.fp16)
            writer.write("max seq length  =%d\n" % args.max_seq_length)
            
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
    return results


def load_and_cache_examples(args, task, tokenizer, evaluate=False, test=False):
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    
    processor = processors[task]()
    # Load data features from cache or dataset file
    if evaluate:
        cached_mode = 'dev'
    elif test:
        cached_mode = 'test'
    else:
        cached_mode = 'train'
    assert (evaluate == True and test == True) == False
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}_{}'.format(
        cached_mode,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length),
        str(task)))
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        logger.info("Creating features from dataset file at %s", args.data_dir)
        label_list = processor.get_labels()
        if evaluate:
            examples = processor.get_dev_examples(args.data_dir)
        elif test:
            examples = processor.get_test_examples(args.data_dir)
        else:
            examples = processor.get_train_examples(args.data_dir)
        
        logger.info("Training number: %s", str(len(examples)))
        
        features = convert_examples_to_features(
            examples,
            label_list,
            
            args.max_seq_length,
            args.max_knowledge_length,
            tokenizer,
            pad_on_left=bool(args.model_type in ['xlnet']),                 # pad on the left for xlnet
            pad_token_segment_id=4 if args.model_type in ['xlnet'] else 0
        )
        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save(features, cached_features_file)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache
    
    # Convert to Tensors and build dataset

    all_input_ids = torch.tensor(select_field(features, 'input_ids'), dtype=torch.long)
    all_input_mask = torch.tensor(select_field(features, 'input_mask'), dtype=torch.long)
    all_segment_ids = torch.tensor(select_field(features, 'segment_ids'), dtype=torch.long)
    all_knowledge_ids = torch.tensor(select_field(features, 'knowledge_ids'), dtype=torch.long)
    all_knowledge_mask = torch.tensor(select_field(features, 'knowledge_mask'), dtype=torch.long)

    

    all_label_ids = torch.tensor([f.label for f in features], dtype=torch.long)
    all_example_ids = torch.tensor([f.example_id for f in features])
     
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_knowledge_ids, all_knowledge_mask, all_label_ids, all_example_ids)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list: " + ", ".join(ALL_MODELS))
    parser.add_argument("--task_name", default=None, type=str, required=True,
                        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()))
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_knowledge_length", default=1024, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true', help='Whether to run test on the test set')
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=50,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError("Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(args.output_dir))

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s", args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args)
    if args.model_name_or_path =="roberta-base": 
          cfg = {"n_head": 4, "attn_pdrop": 0.1, "layer_norm_epsilon": 1e-5, "output_hidden_states":True, "output_attentions":True, "time": 1584474965.1214447, "embd_pdrop": 0.1, "l2": 0.01, "encoder_path": "model/encoder_bpe_40000.json", "n_embd": 768, "resid_pdrop": 0.1, "analysis": True, "afn": "gelu", "vector_l2": False, "lr": 6.25e-05, "opt": "adam", "bpe_path": "model/vocab_40000.bpe", "clf_pdrop": 0.1, "submit": True, "b1": 0.9, "e": 1e-08, "lr_schedule": "warmup_linear", "b2": 0.999, "lr_warmup": 0.002, "lm_coef": 0.5, "seed": 42, "max_grad_norm": 1, "n_layer": 3}
    else:
          cfg = {"n_head": 4, "attn_pdrop": 0.1, "layer_norm_epsilon": 1e-5, "output_hidden_states":True, "output_attentions":True, "time": 1584474965.1214447, "embd_pdrop": 0.1, "l2": 0.01, "encoder_path": "model/encoder_bpe_40000.json", "n_embd": 1024, "resid_pdrop": 0.1, "analysis": True, "afn": "gelu", "vector_l2": False, "lr": 6.25e-05, "opt": "adam", "bpe_path": "model/vocab_40000.bpe", "clf_pdrop": 0.1, "submit": True, "b1": 0.9, "e": 1e-08, "lr_schedule": "warmup_linear", "b2": 0.999, "lr_warmup": 0.002, "lm_coef": 0.5, "seed": 42, "max_grad_norm": 1, "n_layer": 3}
          
    #device = torch.device("cuda") #if torch.cuda.is_available()) #and not args.no_cuda else "cpu")
    #device = args.device
    text_encoder = TextEncoder(cfg["encoder_path"], cfg["bpe_path"])
    encoder = text_encoder.encoder
    encoder['_classify_'] = len(encoder)
    clf_token = encoder['_classify_']
    n_vocab = len(text_encoder.encoder) 
    n_ctx_know = args.max_knowledge_length
    vocab = n_vocab + n_ctx_know

    encode_model = OpenAIGPTDoubleHeadsModel(cfg, clf_token, vocab, n_ctx_know)
    encode_model.to(device)
    encode_model = nn.DataParallel(encode_model) 
    if args.model_name_or_path =="roberta-base": 
         config = {"n_head": 4, "attn_pdrop": 0.1, "layer_norm_epsilon": 1e-5, "output_hidden_states":True, "output_attentions":True, "time": 1584474965.1214447, "embd_pdrop": 0.1, "l2": 0.01, "encoder_path": "model/encoder_bpe_40000.json", "n_embd": 768, "resid_pdrop": 0.1, "analysis": True, "afn": "gelu", "vector_l2": False, "lr": 6.25e-05, "opt": "adam", "bpe_path": "model/vocab_40000.bpe", "clf_pdrop": 0.1, "submit": True, "b1": 0.9, "e": 1e-08, "lr_schedule": "warmup_linear", "b2": 0.999, "lr_warmup": 0.002, "lm_coef": 0.5, "seed": 42, "max_grad_norm": 1, "n_layer": 4}
    else:
         config = {"n_head": 4, "attn_pdrop": 0.1, "layer_norm_epsilon": 1e-5, "output_hidden_states":True, "output_attentions":True, "time": 1584474965.1214447, "embd_pdrop": 0.1, "l2": 0.01, "encoder_path": "model/encoder_bpe_40000.json", "n_embd": 1024, "resid_pdrop": 0.1, "analysis": True, "afn": "gelu", "vector_l2": False, "lr": 6.25e-05, "opt": "adam", "bpe_path": "model/vocab_40000.bpe", "clf_pdrop": 0.1, "submit": True, "b1": 0.9, "e": 1e-08, "lr_schedule": "warmup_linear", "b2": 0.999, "lr_warmup": 0.002, "lm_coef": 0.5, "seed": 42, "max_grad_norm": 1, "n_layer": 4}
    n_ctx = args.max_seq_length
    reasoner_model = DoubleReasonerModel(config, clf_token, vocab, n_ctx, n_ctx_know)
    reasoner_model.to(device)
    reasoner_model = nn.DataParallel(reasoner_model) 
  
    n_gpu = torch.cuda.device_count()
    linear_classifier = Classifier(cfg)
    linear_classifier.to(device)
    linear_classifier = nn.DataParallel(linear_classifier)

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path, num_labels=num_labels, finetuning_task=args.task_name)
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path, do_lower_case=args.do_lower_case)
    model = model_class.from_pretrained(args.model_name_or_path, from_tf=bool('.ckpt' in args.model_name_or_path), config=config)

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)

    logger.info("Training/evaluation parameters %s", args)
    best_steps = 0

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss, best_steps = train(args, train_dataset, model, tokenizer, encode_model, reasoner_model, linear_classifier)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(args.output_dir)

        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)

        tokenizer.save_pretrained(args.output_dir)
        PATH = args.output_dir+"/model.pt"


        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, 'training_args.bin'))
        
        torch.save({'encode_state_dict': encode_model.state_dict(),
                    'reasoner_state_dict': reasoner_model.state_dict()}, PATH)
        # Load a trained model and vocabulary that you have fine-tuned
        model = model_class.from_pretrained(args.output_dir)
        tokenizer = tokenizer_class.from_pretrained(args.output_dir)
        model.to(args.device)
        checkpoint_sub = torch.load(PATH)
        encode_model.load_state_dict(checkpoint_sub['encode_state_dict'])
        reasoner_model.load_state_dict(checkpoint_sub['reasoner_state_dict'])
        
    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        if not args.do_train:
            args.output_dir = args.model_name_or_path
        checkpoints = [args.output_dir]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
            logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            result = evaluate(args, model, encode_model, reasoner_model, linear_classifier, tokenizer, global_step, prefix=prefix)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
            print("DEV Result:", results)

    if args.do_test and args.local_rank in [-1, 0]:
        if not args.do_train:
            args.output_dir = args.model_name_or_path
        checkpoints = [args.output_dir]
        # if args.eval_all_checkpoints: # can not use this to do test!!
        #     checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + '/**/' + WEIGHTS_NAME, recursive=True)))
        #     logging.getLogger("transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        for checkpoint in checkpoints:
            global_step = checkpoint.split('-')[-1] if len(checkpoints) > 1 else ""
            prefix = checkpoint.split('/')[-1] if checkpoint.find('checkpoint') != -1 else ""
            PATH = args.output_dir+"/model.pt"
            model = model_class.from_pretrained(checkpoint)
            model.to(args.device)
            checkpoint_sub = torch.load(PATH)
            encode_model.load_state_dict(checkpoint_sub['encode_state_dict'])
            reasoner_model.load_state_dict(checkpoint_sub['reasoner_state_dict'])
            
            result = evaluate(args, model, encode_model, reasoner_model, linear_classifier, tokenizer, global_step, prefix=prefix, test=True)
            result = dict((k + '_{}'.format(global_step), v) for k, v in result.items())
            results.update(result)
            print("TEST Result:", results)
    if best_steps:
        logger.info("best steps of eval acc is the following checkpoints: %s", best_steps)
    return results


if __name__ == "__main__":
    main()
