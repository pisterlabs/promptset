#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : gpt_double_heads_model.py
# Create date : 2019-03-16 15:25
# Modified date : 2019-03-20 17:01
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

from torch.nn import CrossEntropyLoss
from pybase import pylog

from .gpt.gpt_transformer import GPTTransformer
from .gpt.gpt_pretrained_model import OpenAIGPTPreTrainedModel
from .gpt.base_gpt.model_multiple_choice_head import OpenAIGPTMultipleChoiceHead
from .gpt.base_gpt.model_lm_head import OpenAIGPTLMHead

class OpenAIGPTDoubleHeadsModel(OpenAIGPTPreTrainedModel):
    def __init__(self, config):
        super(OpenAIGPTDoubleHeadsModel, self).__init__(config)
        self.transformer = GPTTransformer(config)
        self.lm_head = OpenAIGPTLMHead(self.transformer.tokens_embed.weight, config)
        self.multiple_choice_head = OpenAIGPTMultipleChoiceHead(config)
        self.apply(self.init_weights)

    def set_num_special_tokens(self, num_special_tokens):
        self.transformer.set_num_special_tokens(num_special_tokens)
        self.lm_head.set_embeddings_weights(self.transformer.tokens_embed.weight)

    def forward(self, input_ids, mc_token_ids, lm_labels=None, mc_labels=None, token_type_ids=None, position_ids=None):
        hidden_states = self.transformer(input_ids, position_ids, token_type_ids)
        lm_logits = self.lm_head(hidden_states)
        mc_logits = self.multiple_choice_head(hidden_states, mc_token_ids)
        losses = []
        if lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            losses.append(loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1)))
        if mc_labels is not None:
            loss_fct = CrossEntropyLoss()
            losses.append(loss_fct(mc_logits.view(-1, mc_logits.size(-1)), mc_labels.view(-1)))
        if losses:
            return losses
        return lm_logits, mc_logits
