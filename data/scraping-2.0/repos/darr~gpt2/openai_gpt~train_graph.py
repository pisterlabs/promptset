#!/usr/bin/python
# -*- coding: utf-8 -*-
#####################################
# File name : train_graph.py
# Create date : 2019-03-15 21:25
# Modified date : 2019-03-22 16:43
# Author : DARREN
# Describe : not set
# Email : lzygzh@126.com
#####################################
from __future__ import division
from __future__ import print_function

from tqdm import tqdm, trange
import torch
from pybase import pylog

from openai_gpt import OpenAIGPTDoubleHeadsModel
from openai_gpt import get_gpt_optimizer

from .down_cache import get_model_file_path
from . import func
from . import rocstories_dataset
from . import show

def _get_pretrained_model(config):
    device = config["device"]
    model_file = get_model_file_path("model", config)
    config_file = get_model_file_path("config", config)
    model = OpenAIGPTDoubleHeadsModel.from_pretrained(model_file, config_file, config)
    model.to(device)
    return model

def _train_the_model(model, train_dataloader, optimizer, config):
    num_train_epochs = config["num_train_epochs"]
    device = config["device"]
    lm_coef = config["lm_coef"]
    nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
    model.train()
    for _ in trange(int(num_train_epochs), desc="Epoch"):
        tr_loss = 0
        nb_tr_steps = 0
        tqdm_bar = tqdm(train_dataloader, desc="Training")
        for step, batch in enumerate(tqdm_bar):
            batch = tuple(t.to(device) for t in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels = batch
            losses = model(input_ids, mc_token_ids, lm_labels, mc_labels)
            loss = lm_coef * losses[0] + losses[1]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            tr_loss += loss.item()
            exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
            nb_tr_steps += 1
            tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, optimizer.get_lr()[0])
    train_loss = tr_loss/nb_tr_steps if config["do_train"] else None
    return train_loss

def _save_finetuned_model(model, config):
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    state_dict = model_to_save.state_dict()
#    show.show_model_paramter_size(state_dict)

    output_model_file = func.get_output_model_file_full_path(config)
    torch.save(model_to_save.state_dict(), output_model_file)

def do_train(config):
    func.init_app(config)
    model = _get_pretrained_model(config)
    train_dataloader, train_data = rocstories_dataset.get_train_dataloader(model, config)
    optimizer = get_gpt_optimizer(model, train_data, config)

    train_loss = _train_the_model(model, train_dataloader, optimizer, config)
    model_config = model.config
    _save_finetuned_model(model, config)

    return train_loss, model_config

def direct_save(config):
    func.init_app(config)
    model = _get_pretrained_model(config)
    model_config = model.config
    _save_finetuned_model(model, config)
    return 0.0, model_config
