#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : gpt_pretrained_model.py
# Create date : 2019-03-16 14:33
# Modified date : 2019-03-22 16:41
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn

from pybase import pylog

from .base_gpt.model_base import LayerNorm
from .base_gpt.load_model import change_state_dict
from .base_gpt.load_model import show_msg
from .gpt_config import OpenAIGPTConfig

def show_model_paramter_size(state_dict):
    count = 0
    for key in state_dict:
        pylog.info(key)
        pylog.info(state_dict[key].size())
        size_lt = state_dict[key].size()
        size_num = 1
        for i in size_lt:
            size_num *= i
        count += size_num

    pylog.info("paramete count:%d" % count)
    pylog.info("paramete count :%d bytes" % (count * 4))
    pylog.info("paramete count :%d M" % ((count * 4) / (1024* 1024)))

def load_state_dict_to_model(module, state_dict, missing_keys, unexpected_keys, error_msgs, prefix=""):
    metadata = state_dict._metadata
    local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
    module._load_from_state_dict(state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
    for name, child in module._modules.items():
        if child is not None:
            load_state_dict_to_model(child, state_dict, missing_keys, unexpected_keys, error_msgs, prefix + name + ".")

def load_model(module, state_dict, class_name, prefix=""):
    missing_keys = []
    unexpected_keys = []
    error_msgs = []

    metadata = state_dict._metadata
    load_state_dict_to_model(module, state_dict, missing_keys, unexpected_keys, error_msgs, prefix="")
    show_msg(class_name, missing_keys, unexpected_keys, error_msgs)

def _check_model(model, state_dict):
    start_model = model

    if hasattr(model, "transformer") and all(not s.startswith('transformer.') for s in state_dict.keys()):
        start_model = model.transformer
    return start_model

def _copy_state_dict(state_dict):
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata
    return state_dict

def _load_state_dict_from_file(model_file):
    state_dict = torch.load(model_file, map_location='cpu' if not torch.cuda.is_available() else None)
    change_state_dict(state_dict)
    return state_dict

class OpenAIGPTPreTrainedModel(nn.Module):
    """
        An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    def __init__(self, config, *inputs, **kwargs):
        super(OpenAIGPTPreTrainedModel, self).__init__()
        if not isinstance(config, OpenAIGPTConfig):
            pylog.error("config should be an istance of class OpenAIGPTConfig")
        self.config = config

    def init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def set_num_special_tokens(self, num_special_tokens):
        # Add additional embeddings for special tokens if needed
        # This step also make sure we are still sharing the output and input embeddings after loading weights
        pass

    @classmethod
    def from_pretrained_tf(cls, model_file, config_file, config, *inputs, **kwargs):
        return cls._load_model_from_tf(model_file, config_file, config, *inputs, **kwargs)

    @classmethod
    def from_pretrained(cls, model_file, config_file, config, *inputs, **kwargs):
        """
        Instantiate a OpenAIGPTPreTrainedModel from a pre-trained model file or a pytorch state dict.
        """
        return cls._load_model_from_troch(model_file, config_file, config, *inputs, **kwargs)

    @classmethod
    def _load_model_from_troch(cls, model_file, config_file, config, *inputs, **kwargs):
        state_dict = _load_state_dict_from_file(model_file)
        state_dict = _copy_state_dict(state_dict)

        model_config = OpenAIGPTConfig.from_json_file(config_file)
        model = cls(model_config, *inputs, **kwargs)

        start_model = _check_model(model, state_dict)
        class_name = model.__class__.__name__
        load_model(start_model, state_dict, class_name, prefix="")

        num_special_tokens = len(config["special_tokens_lt"])
        model.set_num_special_tokens(num_special_tokens)

        return model

    @classmethod
    def _load_model_from_tf(cls, model_file, config_file, config, *inputs, **kwargs):
        model_config = OpenAIGPTConfig.from_json_file(config_file)
        model = cls(model_config, *inputs, **kwargs)

        # Directly load from a TensorFlow checkpoint (stored as NumPy array)
        return load_tf_weights_in_openai_gpt(model, model_file)
