#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : gpt_transformer.py
# Create date : 2019-03-16 15:10
# Modified date : 2019-03-22 16:42
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import copy
import torch
import torch.nn as nn

from .base_gpt.model_block import Block
from .gpt_pretrained_model import OpenAIGPTPreTrainedModel

class GPTTransformer(OpenAIGPTPreTrainedModel):
    def __init__(self, config):
        super(GPTTransformer, self).__init__(config)
        num_tokens = config.vocab_size + config.n_special
        self.tokens_embed = nn.Embedding(num_tokens, config.n_embd)
        self.positions_embed = nn.Embedding(config.n_positions, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        block = Block(config.n_ctx, config, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(config.n_layer)])

        self.apply(self.init_weights)
        # nn.init.normal_(self.embed.weight, std=0.02)

    def set_num_special_tokens(self, num_special_tokens):
        " Update input embeddings with new embedding matrice if needed "
        if self.config.n_special == num_special_tokens:
            return
        # Update config
        self.config.n_special = num_special_tokens
        # # Build new embeddings and initialize
        old_embed = self.tokens_embed
        self.tokens_embed = nn.Embedding(self.config.total_tokens_embeddings, self.config.n_embd)
        # Initialize all new embeddings (in particular the special tokens)
        self.init_weights(self.tokens_embed)
        # Copy word and positional embeddings from the previous weights
        self.tokens_embed.weight.data[: self.config.vocab_size, :] = old_embed.weight.data[: self.config.vocab_size, :]
        self.tokens_embed.weight.data[-self.config.n_positions :, :] = old_embed.weight.data[-self.config.n_positions :, :]

    def forward(self, input_ids, position_ids=None, token_type_ids=None):
        if position_ids is None:
            # This was used when we had a single embedding matrice from position and token embeddings
            # start = self.config.vocab_size + self.config.n_special
            # end = start + input_ids.size(-1)
            # position_ids = torch.arange(start, end, dtype=torch.long, device=input_ids.device)
            position_ids = torch.arange(input_ids.size(-1), dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        inputs_embeds = self.tokens_embed(input_ids)
        position_embeds = self.positions_embed(position_ids)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1))
            token_type_embeds = self.tokens_embed(token_type_ids)
        else:
            token_type_embeds = 0
        # Add the position information to the input embeddings
        # h = e.sum(dim=2)
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        for block in self.h:
            hidden_states = block(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        return hidden_states.view(*output_shape)
