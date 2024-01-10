#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : rocstories_dataset_test.py
# Create date : 2019-03-17 12:52
# Modified date : 2019-03-17 15:57
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

import sys
sys.path.append("..")
import etc
from etc import config
from pybase import pylog
import rocstories_dataset
from openai_gpt import OpenAIGPTTokenizer

def _test_load_rocstories_dataset():
    train_dataset = rocstories_dataset._load_rocstories_dataset(config["train_dataset"])
    print(len(train_dataset))

def _test_load_catched_data():
    rocstories_dataset._load_catched_data(config)

def _test_tokenize_and_encode():
    pass
#   obj 
#   tokenizer
#   ret = rocstories_dataset._tokenize_and_encode(obj, tokenizer)

def _test_get_encoded_datasets():
    special_tokens = config["special_tokens"]
    tokenizer = OpenAIGPTTokenizer.from_pretrained(config["model_name"], special_tokens=special_tokens)
    print(tokenizer)
    #encoded_datasets = rocstories_dataset._get_encoded_datasets(tokenizer, config)

def test():
    _test_load_rocstories_dataset()
    _test_load_catched_data()
    #_test_get_encoded_datasets()
