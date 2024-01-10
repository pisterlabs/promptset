import argparse
from nltk.tokenize import sent_tokenize

import re
import dialog_config

# from AgentProfile.profiles_in_dev import GlobalProfile
import torch.nn as nn
from utils import print_cm
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import regex as re
import random
import itertools
import tqdm
import time

import warnings
import signal
import sys
from sys import exit

warnings.filterwarnings("ignore")
import pdb
import os
import csv
import pickle as pkl

# from gpt_model import GPT2SimpleLM
# from pytorch_pretrained_bert import GPT2Tokenizer
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from GPTModel1 import GPT2LMHeadModel_modified
from pytorch_pretrained_bert import OpenAIAdam
import config as cfg
from torch.nn import Identity
from utils import is_repetition_with_context

from KnowledgeBase.KB import HumanRule
from KnowledgeBase import KB
from KnowledgeBase.KB import Domain
from nltk.tokenize import sent_tokenize
import logging

from copy import deepcopy
import tqdm
import pandas as pd


def load_model(cfg, model_size, tokenizer, device1, device2):
    if model_size == "small":
        model_A = GPT2LMHeadModel_modified.from_pretrained(
            "gpt2", output_hidden_states=True
        )
        model_B = GPT2LMHeadModel_modified.from_pretrained(
            "gpt2", output_hidden_states=True
        )
    elif model_size == "medium":
        model_A = GPT2LMHeadModel_modified.from_pretrained(
            "gpt2-medium", output_hidden_states=True
        )
        model_B = GPT2LMHeadModel_modified.from_pretrained(
            "gpt2-medium", output_hidden_states=True
        )

    model_A.resize_token_embeddings(len(tokenizer))
    model_B.resize_token_embeddings(len(tokenizer))

    # load the model
    if False:
        if cfg.model_size == "small":
            model_A_states, model_B_states = torch.load(cfg.new_small_model_dir)
        elif cfg.model_size == "medium":
            if cfg.use_old_model:
                model_A_states, model_B_states = torch.load(cfg.old_medium_model_dir)
                model_A_states["transformer.wte.weight"] = model_A_states[
                    "transformer.wte.weight"
                ][:50257, :]
                model_B_states["transformer.wte.weight"] = model_B_states[
                    "transformer.wte.weight"
                ][:50257, :]
            else:
                model_A_states, model_B_states = torch.load(cfg.new_medium_model_dir)

        model_A.load_state_dict(model_A_states, strict=False)
        model_B.load_state_dict(model_B_states, strict=False)
    else:
        # initialize
        pass

    model_A = model_A.to(device1)
    model_B = model_B.to(device2)
    model_A.device = device1
    model_B.device = device2

    return model_A, model_B


def load_pkl(dir):
    with open(dir, "rb") as fh:
        obj = pkl.load(fh)
    return obj


def save_pkl(obj, dir):
    with open(dir, "wb") as fh:
        pkl.dump(obj, fh)


def split_train_val():
    import pickle as pkl

    df = pd.read_excel("training/data/300_dialog.xlsx")
    df = df[(df["er_label_1"] != "off-task") & (df["er_label_1"] != "other")]
    df = df[(df["ee_label_1"] != "off-task") & (df["ee_label_1"] != "other")]
    from sklearn import preprocessing

    le_A = preprocessing.LabelEncoder()
    labels_A = le_A.fit_transform(df[df["B4"] == 0]["er_label_1"])
    df["er_label_1_num"] = None  # labels
    df["er_label_1_num"].loc[df["B4"] == 0] = labels_A
    save_pkl(le_A, "training/data/labelencoder_A.pkl")
    le_B = preprocessing.LabelEncoder()
    labels_B = le_B.fit_transform(df[df["B4"] == 1]["ee_label_1"])
    df["ee_label_1_num"] = None  # labels
    df["ee_label_1_num"].loc[df["B4"] == 1] = labels_B
    save_pkl(le_B, "training/data/labelencoder_B.pkl")

    def extract_data(df_dialogs):
        data = {}
        prev_dialog_id = "init"
        # prev_role_id = 1
        for i in tqdm.trange(len(df_dialogs)):
            line = df.iloc[i]
            cur_dialog_id, cur_role_id = line["B2"], line["B4"]
            if cur_dialog_id not in data:
                data[cur_dialog_id] = []
            if cur_dialog_id == prev_dialog_id:
                # the same dialog
                # import pdb
                # pdb.set_trace()
                if prev_role_id != cur_role_id:
                    # different person talking
                    if line["B4"] == 0:
                        data[cur_dialog_id].append([cur_sents, cur_acts, cur_roles])
                        text = "A:" + line["Unit"].strip()
                        cur_sents = [text]
                        cur_acts = [line["er_label_1_num"]]
                        cur_roles = [0]
                    else:
                        data[cur_dialog_id].append([cur_sents, cur_acts, cur_roles])
                        text = "B:" + line["Unit"].strip()
                        cur_sents = [text]
                        cur_acts = [line["ee_label_1_num"]]
                        cur_roles = [1]
                else:
                    # the same person
                    text = line["Unit"].strip()
                    if line["B4"] == 0:
                        cur_sents.append(text)
                        cur_acts.append(line["er_label_1_num"])
                        cur_roles.append(0)
                    else:
                        cur_sents.append(text)
                        cur_acts.append(line["ee_label_1_num"])
                        cur_roles.append(1)
            else:
                # a new dialog
                if prev_dialog_id != "init":
                    data[prev_dialog_id].append([cur_sents, cur_acts, cur_roles])
                if line["B4"] == 0:
                    text = "A:" + line["Unit"].strip()
                    cur_sents = [text]
                    cur_acts = [line["er_label_1_num"]]
                    cur_roles = [0]
                else:
                    text = "B:" + line["Unit"].strip()
                    cur_sents = [text]
                    cur_acts = [line["ee_label_1_num"]]
                    cur_roles = [1]
            prev_dialog_id = cur_dialog_id
            prev_role_id = cur_role_id
        return data

    all_data = extract_data(df)
    import random

    random.seed(123)
    ids = list(range(len(all_data)))
    random.shuffle(ids)
    all_data = [list(all_data.values())[i] for i in ids]
    train_data = all_data[: int(len(all_data) * 0.85)]
    val_data = all_data[int(len(all_data) * 0.85) :]
    save_pkl(train_data, "training/data/train_data.pkl")
    save_pkl(val_data, "training/data/val_data.pkl")


def split_train_val_TF():
    import pickle as pkl

    data = torch.load(
        "demonstration/old_model/demonstration_torchsave.pkl", map_location="cuda:0"
    )

    contexts = []
    sents = []
    ys = []
    for d in data:
        for candidate in d["individual_features"]:
            contexts.append(d["shared_features"]["context"])
            sents.append(candidate["sent"])
            ys.append(candidate["pick_or_not"])
    all_data = list(zip(contexts, sents, ys))

    import random

    random.seed(123)
    random.shuffle(all_data)
    train_data = all_data[: int(len(all_data) * 0.8)]
    val_data = all_data[int(len(all_data) * 0.8) :]

    torch.save(
        train_data, "demonstration/old_model/demonstration_train_with_text_only.pkl"
    )
    torch.save(val_data, "demonstration/old_model/demonstration_val_with_text_only.pkl")


class SequenceSummary(nn.Module):
    r"""Compute a single vector summary of a sequence hidden states according to various possibilities:
    Args of the config class:
        summary_type:
            - 'last' => [default] take the last token hidden state (like XLNet)
            - 'first' => take the first token hidden state (like Bert)
            - 'mean' => take the mean of all tokens hidden states
            - 'cls_index' => supply a Tensor of classification token position (GPT/GPT-2)
            - 'attn' => Not implemented now, use multi-head attention
        summary_use_proj: Add a projection after the vector extraction
        summary_proj_to_labels: If True, the projection outputs to config.num_labels classes (otherwise to hidden_size). Default: False.
        summary_activation: 'tanh' => add a tanh activation to the output, Other => no activation. Default
        summary_first_dropout: Add a dropout before the projection and activation
        summary_last_dropout: Add a dropout after the projection and activation
    """

    def __init__(self, num_labels, config):
        super().__init__()
        self.config = config
        self.summary_type = (
            config.summary_type if hasattr(config, "summary_type") else "last"
        )
        if self.summary_type == "attn":
            # We should use a standard multi-head attention module with absolute positional embedding for that.
            # Cf. https://github.com/zihangdai/xlnet/blob/master/modeling.py#L253-L276
            # We can probably just use the multi-head attention module of PyTorch >=1.1.0
            raise NotImplementedError
        self.summary = Identity()
        if hasattr(config, "summary_use_proj") and config.summary_use_proj:
            if (
                hasattr(config, "summary_proj_to_labels")
                and config.summary_proj_to_labels
                and num_labels > 0
            ):
                print(f"num_class: {num_labels}")
                num_classes = num_labels
            else:
                print(f"num_class here: {config.hidden_size}")
                num_classes = config.hidden_size
            self.summary = nn.Linear(config.hidden_size, num_classes)
        self.activation = nn.Tanh()
        if (
            hasattr(config, "summary_activation")
            and config.summary_activation == "tanh"
        ):
            self.activation = nn.Tanh()
        self.first_dropout = Identity()
        if (
            hasattr(config, "summary_first_dropout")
            and config.summary_first_dropout > 0
        ):
            self.first_dropout = nn.Dropout(config.summary_first_dropout)
        self.last_dropout = Identity()
        if hasattr(config, "summary_last_dropout") and config.summary_last_dropout > 0:
            self.last_dropout = nn.Dropout(config.summary_last_dropout)

        self.apply(self.init_weights)

    def init_weights(self, module):
        """Initialize the weights."""
        from transformers.modeling_gpt2 import Conv1D

        if isinstance(module, (nn.Linear, nn.Embedding, Conv1D)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if isinstance(module, (nn.Linear, Conv1D)) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, hidden_states, cls_index=None):
        """hidden_states: float Tensor in shape [bsz, ..., seq_len, hidden_size], the hidden-states of the last layer.
        cls_index: [optional] position of the classification token if summary_type == 'cls_index',
            shape (bsz,) or more generally (bsz, ...) where ... are optional leading dimensions of hidden_states.
            if summary_type == 'cls_index' and cls_index is None:
                we take the last token of the sequence as classification token
        """
        if self.summary_type == "last":
            output = hidden_states[:, -1]
        elif self.summary_type == "first":
            output = hidden_states[:, 0]
        elif self.summary_type == "mean":
            output = hidden_states.mean(dim=1)
        elif self.summary_type == "cls_index":
            if cls_index is None:
                cls_index = torch.full_like(
                    hidden_states[..., :1, :],
                    hidden_states.shape[-2] - 1,
                    dtype=torch.long,
                )
            else:
                cls_index = cls_index.unsqueeze(-1).unsqueeze(-1)
                cls_index = cls_index.expand(
                    (-1,) * (cls_index.dim() - 1) + (hidden_states.size(-1),)
                )
            # shape of cls_index: (bsz, XX, 1, hidden_size) where XX are optional leading dim of hidden_states
            output = hidden_states.gather(-2, cls_index).squeeze(
                -2
            )  # shape (bsz, XX, hidden_size)
        elif self.summary_type == "attn":
            raise NotImplementedError
        output = self.first_dropout(output)
        output = self.summary(output)
        output = self.activation(output)
        output = self.last_dropout(output)
        return output


class PersuadeDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.tokenizer.max_len = 1500
        # tokenizer weird behavior
        self.cls_token_id = tokenizer.cls_token_id  # [628, 198]
        self.turn_ending = [628, 198]
        # tokenizer.encode("\n\n\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # dial_tokens = [tokenizer.encode(item) + self.turn_ending for item in self.data[index]]
        role_ids = [turn[2][0] for turn in self.data[index]]
        whole_sents = []
        for turn in self.data[index]:
            temp_sent = turn[0]
            temp_sent = " ".join(temp_sent)
            try:
                assert temp_sent.startswith("A:") or temp_sent.startswith("B:")
            except:
                pdb.set_trace()
            whole_sents.append(
                self.tokenizer.encode(temp_sent[:2])
                + self.tokenizer.encode(temp_sent[2:])
                + self.turn_ending
            )
            # assert whole_sents[-1]
        separate_sents = []
        for turn in self.data[index]:
            turn_sents = []
            for i, sent in enumerate(turn[0]):
                if i == 0:
                    turn_sents.append(
                        self.tokenizer.encode(sent[:2])
                        + self.tokenizer.encode(sent[2:])
                    )
                else:
                    turn_sents.append(self.tokenizer.encode(" " + sent))
            separate_sents.append(turn_sents)
        acts = []
        for turn in self.data[index]:
            turn_acts = []
            for i, act in enumerate(turn[1]):
                turn_acts.append(act)
            acts.append(turn_acts)
        return role_ids, whole_sents, separate_sents, acts

    def collate(self, unpacked_data):
        return unpacked_data


class TFDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.tokenizer.max_len = 1500
        # tokenizer weird behavior
        self.cls_token_id = tokenizer.cls_token_id  # [628, 198]
        self.turn_ending = [628, 198]
        # tokenizer.encode("\n\n\n")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        # dial_tokens = [tokenizer.encode(item) + self.turn_ending for item in self.data[index]]
        context = [
            "A:" + " ".join(s) if i % 2 == 0 else "B:" + " ".join(s)
            for i, s in enumerate(self.data[index][0])
        ]
        context = [" ".join(s) for i, s in enumerate(self.data[index][0])]
        context = [
            self.tokenizer.encode("A:") + self.tokenizer.encode(c) + self.turn_ending
            if i % 2 == 0
            else self.tokenizer.encode("B:")
            + self.tokenizer.encode(c)
            + self.turn_ending
            for i, c in enumerate(context)
        ]
        candidate_sent = self.tokenizer.encode("A:") + self.tokenizer.encode(
            " ".join(self.data[index][1])
        )
        y = int(self.data[index][2])
        return context, candidate_sent, y

    def collate(self, unpacked_data):
        return unpacked_data


class ModelClassifier(object):
    def __init__(
        self,
        config,
        which_to_train,
        model_A,
        model_B,
        tokenizer,
        device1,
        device2,
        clf_A=None,
        clf_B=None,
        clf_TF=None,
    ):
        # config.num_labels = le.classes_.shape[0]
        # label encode
        # super().__init__()
        self.config = config
        self.le_A = load_pkl("training/data/labelencoder_A.pkl")
        self.le_B = load_pkl("training/data/labelencoder_B.pkl")

        self.clf_A = (
            SequenceSummary(num_labels=self.le_A.classes_.shape[0], config=config)
            if clf_A is None
            else clf_A
        )
        self.clf_B = (
            SequenceSummary(num_labels=self.le_B.classes_.shape[0], config=config)
            if clf_B is None
            else clf_B
        )
        self.clf_TF = (
            SequenceSummary(num_labels=2, config=config) if clf_TF is None else clf_TF
        )

        # self.apply(self.init_weight)
        self.past = None
        self.history = []

        # model
        self.model_A = model_A
        self.model_B = model_B
        self.tokenizer = tokenizer
        self.cls_token_id = tokenizer.cls_token_id
        self.device1 = device1
        self.device2 = device2

        self.to_device(self.device1)

        # define loss
        self.criterion = nn.CrossEntropyLoss()

        # optimizer parameters
        self.num_gradients_accumulation = 1
        self.batch_size = 1
        self.batch_size_TF = 8

        # load training data
        self.load_data()

    def reload(self):
        self.past = None
        self.history = []

    def parameters(self):
        parameters_list = (
            list(self.model_A.parameters())
            + list(self.model_B.parameters())
            + list(self.clf_A.parameters())
            + list(self.clf_B.parameters())
            + list(self.clf_TF.parameters())
        )
        return parameters_list

    def to_device(self, device):
        # to device
        self.clf_A = self.clf_A.to(device)
        self.clf_B = self.clf_B.to(device)
        self.clf_TF = self.clf_TF.to(device)

        self.clf_A.device = device
        self.clf_B.device = device
        self.clf_TF.device = device
        # self.model_A = self.model_A.to(self.device)
        # self.model_B = self.model_B.to(self.device)

    def load_data(self):
        # load training data
        self.train_data = load_pkl("training/data/train_data.pkl")
        self.val_data = load_pkl("training/data/val_data.pkl")
        self.train_data_TF, self.val_data_TF = torch.load(
            "demonstration/old_model/demonstration_train_with_text_only.pkl",
            map_location="cpu",
        ), torch.load(
            "demonstration/old_model/demonstration_val_with_text_only.pkl",
            map_location="cpu",
        )

        self.train_dataset = PersuadeDataset(self.train_data, self.tokenizer)
        self.val_dataset = PersuadeDataset(self.val_data, self.tokenizer)

        self.train_dataset_TF, self.val_dataset_TF = TFDataset(
            self.train_data_TF, self.tokenizer
        ), TFDataset(self.val_data_TF, self.tokenizer)

        self.train_dataloader = DataLoader(
            dataset=self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            collate_fn=self.train_dataset.collate,
        )
        self.val_dataloader = DataLoader(
            dataset=self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size,
            collate_fn=self.train_dataset.collate,
        )

        self.train_dataloader_TF = DataLoader(
            dataset=self.train_dataset_TF,
            shuffle=True,
            batch_size=self.batch_size_TF,
            collate_fn=self.train_dataset_TF.collate,
        )
        self.val_dataloader_TF = DataLoader(
            dataset=self.val_dataset_TF,
            shuffle=False,
            batch_size=self.batch_size_TF,
            collate_fn=self.val_dataset_TF.collate,
        )

    def load_model(
        self, all_model_dir=None, clf_A_dir=None, clf_B_dir=None, clf_TF_dir=None
    ):
        if all_model_dir is None:
            if clf_A_dir:
                clf_A_state = torch.load(clf_A_dir)
                self.clf_A.load_state_dict(clf_A_state)
                print(f"clf_A loaded")

            if clf_B_dir:
                clf_B_state = torch.load(clf_B_dir)
                self.clf_B.load_state_dict(clf_B_state)
                print(f"clf_B loaded")

            if clf_TF_dir:
                clf_TF_state = torch.load(clf_TF_dir)
                self.clf_TF.load_state_dict(clf_TF_state)
                print(f"clf_TF loaded")
        else:
            (
                model_A_state,
                model_B_state,
                clf_A_state,
                clf_B_state,
                clf_TF_state,
            ) = torch.load(all_model_dir)
            self.model_A.load_state_dict(model_A_state)
            self.model_B.load_state_dict(model_B_state)
            self.clf_A.load_state_dict(clf_A_state)
            self.clf_B.load_state_dict(clf_B_state)
            self.clf_TF.load_state_dict(clf_TF_state)
            print(f"all models loaded")

    def train(self, which_to_train, num_epochs=10):
        # optimizer
        param_optimizer = list(self.model_A.named_parameters()) + list(
            self.model_B.named_parameters()
        )
        if "A" in which_to_train:
            print("clf_A to optimize")
            param_optimizer += list(self.clf_A.named_parameters())
        if "B" in which_to_train:
            print("clf_B to optimize")
            param_optimizer += list(self.clf_B.named_parameters())
        if "TF" in which_to_train:
            print("clf_TF to optimize")
            param_optimizer += list(self.clf_TF.named_parameters())

        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        num_train_optimization_steps = (
            len(self.train_dataset)
            * num_epochs
            // self.batch_size
            // self.num_gradients_accumulation
        )

        self.optimizer = OpenAIAdam(
            optimizer_grouped_parameters,
            lr=2e-5,
            warmup=0.1,
            max_grad_norm=1.0,
            weight_decay=0.01,
            t_total=num_train_optimization_steps,
        )

        update_count = 0
        progress_bar = tqdm.tqdm
        start = time.time()
        best_acc_A = -float("Inf")
        best_f1_A = -float("Inf")
        best_acc_B = -float("Inf")
        best_f1_B = -float("Inf")
        best_acc_TF = -float("Inf")
        best_f1_TF = -float("Inf")

        for ep in tqdm.tqdm(range(num_epochs)):

            # set train mode
            self.model_A.train()
            self.model_B.train()
            self.clf_A.train()
            self.clf_B.train()
            self.clf_TF.train()

            "Training"
            pbar = progress_bar(self.train_dataloader)
            train_dataloader_TF_list = list(self.train_dataloader_TF)

            for i, batch in enumerate(pbar):
                batch = batch[0]
                batch_TF = train_dataloader_TF_list[i % len(train_dataloader_TF_list)]
                # without relative position
                # if sum([len(item) for item in batch[1]]) > 1024:
                #     input("1024 here!")
                #     continue

                record_loss = self.train_one_iter(
                    batch, batch_TF, update_count, which_to_train, fp16=False
                )
                update_count += 1

                if (
                    update_count % self.num_gradients_accumulation
                    == self.num_gradients_accumulation - 1
                ):
                    # update for gradient accumulation
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    # speed measure
                    end = time.time()
                    speed = (
                        self.batch_size
                        * self.num_gradients_accumulation
                        / (end - start)
                    )
                    start = end

                    # show progress
                    pbar.set_postfix(loss=record_loss, speed=speed)

            "Evaluation"
            self.model_A.eval()
            self.model_B.eval()
            self.clf_A.eval()
            self.clf_B.eval()
            self.clf_TF.eval()

            (
                (val_acc_A, val_f1_A),
                (val_acc_B, val_f1_B),
                (val_acc_TF, val_f1_TF),
            ) = self.validate(
                self.val_dataloader, self.val_dataloader_TF, ep, which_to_train
            )
            print(f"A: val f1: {val_f1_A}, valid acc: {val_acc_A}")
            print(f"B: val f1: {val_f1_B}, valid acc: {val_acc_B}")
            print(f"TF: val f1: {val_f1_TF}, valid acc: {val_acc_TF}")
            is_best_so_far_TF = val_f1_TF > best_f1_TF
            is_best_so_far_A = val_f1_A > best_f1_A
            is_best_so_far_B = val_f1_TF > best_f1_B

            if is_best_so_far_TF:
                best_acc_TF = val_acc_TF
                best_f1_TF = val_f1_TF
            if is_best_so_far_A:
                best_acc_A = val_acc_A
                best_f1_A = val_f1_A
            if is_best_so_far_B:
                best_acc_B = val_acc_B
                best_f1_B = val_f1_B
            SAVED = False
            if is_best_so_far_TF and not SAVED:
                SAVED = True
                torch.save(
                    (
                        self.model_A.state_dict(),
                        self.model_B.state_dict(),
                        self.clf_A.state_dict(),
                        self.clf_B.state_dict(),
                        self.clf_TF.state_dict(),
                    ),
                    f"Checkpoint_act_clf/epoch{ep}_multitask_TF_best_acc_{val_acc_TF}_f1_{val_f1_TF}_A_acc_{val_acc_A}_f1_{val_f1_A}_B_acc_{val_acc_B}_f1_{val_f1_B}.pth",
                )
            if is_best_so_far_A and not SAVED:
                SAVED = True
                torch.save(
                    (
                        self.model_A.state_dict(),
                        self.model_B.state_dict(),
                        self.clf_A.state_dict(),
                        self.clf_B.state_dict(),
                        self.clf_TF.state_dict(),
                    ),
                    f"Checkpoint_act_clf/epoch{ep}_multitask_TF_best_acc_{val_acc_TF}_f1_{val_f1_TF}_A_acc_{val_acc_A}_f1_{val_f1_A}_B_acc_{val_acc_B}_f1_{val_f1_B}.pth",
                )
            if is_best_so_far_B and not SAVED:
                SAVED = True
                torch.save(
                    (
                        self.model_A.state_dict(),
                        self.model_B.state_dict(),
                        self.clf_A.state_dict(),
                        self.clf_B.state_dict(),
                        self.clf_TF.state_dict(),
                    ),
                    f"Checkpoint_act_clf/epoch{ep}_multitask_TF_best_acc_{val_acc_TF}_f1_{val_f1_TF}_A_acc_{val_acc_A}_f1_{val_f1_A}_B_acc_{val_acc_B}_f1_{val_f1_B}.pth",
                )
                # if which_to_train == "A":
                #     torch.save(model_A.state_dict(), f"Checkpoint_act_clf/A/best_acc_{best_acc}_f1_{best_f1}.pth")
                # elif which_to_train == "B":
                #     torch.save(model_A.state_dict(), f"Checkpoint_act_clf/B/best_acc_{best_acc}_f1_{best_f1}.pth")
                # checkpointer.save_checkpoint(ep, model_A.state_dict(), {"None": None}, is_best_so_far)

        print("finally")
        print("A: \nbest acc: {}, best f1: {}".format(best_acc_A, best_f1_A))
        print("B: \nbest acc: {}, best f1: {}".format(best_acc_B, best_f1_B))
        print("TF: \nbest acc: {}, best f1: {}".format(best_acc_TF, best_f1_TF))

    def validate(self, dataloader, dataloader_TF, ep, which_to_train):
        from sklearn.metrics import f1_score
        from sklearn.metrics import confusion_matrix
        from utils import print_cm

        # evaluation mode
        self.model_A.eval()
        self.model_B.eval()
        self.clf_A.eval()
        self.clf_B.eval()
        self.clf_TF.eval()

        def get_numbers_for_one_task(
            sents, logits, acts, x, y_true, y_pred, total, correct
        ):
            _, predicted_acts = torch.max(logits, 1)

            x.extend(sents)
            y_true.extend(acts.tolist()[0])
            y_pred.extend(predicted_acts.tolist())

            total += len(acts.tolist()[0])
            correct += (predicted_acts == acts).sum().item()

            return x, y_true, y_pred, total, correct

        progress_bar = tqdm.tqdm

        with torch.no_grad():
            pbar = progress_bar(dataloader)
            dataloader_TF_list = list(dataloader_TF)
            correct = 0
            total = 0
            x_A, y_true_A, y_pred_A, correct_A, total_A = [], [], [], 0, 0
            x_B, y_true_B, y_pred_B, correct_B, total_B = [], [], [], 0, 0
            x_TF, y_true_TF, y_pred_TF, correct_TF, total_TF = [], [], [], 0, 0

            for i, batch in enumerate(pbar):
                batch = batch[0]
                batch_TF = dataloader_TF_list[i % len(dataloader_TF_list)]
                # if sum([len(item) for item in batch[1]]) > 1024:
                #     continue

                (
                    sents_A,
                    logits_A,
                    acts_A,
                    sents_B,
                    logits_B,
                    acts_B,
                    sents_TF,
                    logits_TF,
                    acts_TF,
                ) = self.train_one_iter(
                    batch,
                    batch_TF,
                    None,
                    which_to_train,
                    fp16=False,
                    is_validation=True,
                )

                x_A, y_true_A, y_pred_A, total_A, correct_A = get_numbers_for_one_task(
                    sents_A,
                    logits_A,
                    acts_A,
                    x_A,
                    y_true_A,
                    y_pred_A,
                    total_A,
                    correct_A,
                )
                x_B, y_true_B, y_pred_B, total_B, correct_B = get_numbers_for_one_task(
                    sents_B,
                    logits_B,
                    acts_B,
                    x_B,
                    y_true_B,
                    y_pred_B,
                    total_B,
                    correct_B,
                )
                (
                    x_TF,
                    y_true_TF,
                    y_pred_TF,
                    total_TF,
                    correct_TF,
                ) = get_numbers_for_one_task(
                    sents_TF,
                    logits_TF,
                    acts_TF,
                    x_TF,
                    y_true_TF,
                    y_pred_TF,
                    total_TF,
                    correct_TF,
                )

            f1_A = f1_score(y_true_A, y_pred_A, average="weighted")
            f1_B = f1_score(y_true_B, y_pred_B, average="weighted")
            f1_TF = f1_score(y_true_TF, y_pred_TF, average="binary")
            # pdb.set_trace()

            pd.DataFrame(
                zip(
                    x_A,
                    self.le_A.inverse_transform(y_true_A).tolist(),
                    self.le_A.inverse_transform(y_pred_A).tolist(),
                ),
                columns=["sent", "y_true", "y_pred"],
            ).to_csv(
                f"Checkpoint_act_clf/A/act_classifier_val_results_epoch{ep}.csv",
                index=None,
            )
            print(f"A: Epoch {ep} Validation accuracy: {correct_A/total_A}, f1: {f1_A}")

            pd.DataFrame(
                zip(
                    x_B,
                    self.le_B.inverse_transform(y_true_B).tolist(),
                    self.le_B.inverse_transform(y_pred_B).tolist(),
                ),
                columns=["sent", "y_true", "y_pred"],
            ).to_csv(
                f"Checkpoint_act_clf/B/act_classifier_val_results_epoch{ep}.csv",
                index=None,
            )
            print(f"B: Epoch {ep} Validation accuracy: {correct_B/total_B}, f1: {f1_B}")

            pd.DataFrame(
                zip(x_TF, y_true_TF, y_pred_TF), columns=["sent", "y_true", "y_pred"]
            ).to_csv(
                f"Checkpoint_act_clf/TF/act_classifier_val_results_epoch{ep}.csv",
                index=None,
            )
            print(
                f"TF: Epoch {ep} Validation accuracy: {correct_TF/total_TF}, f1: {f1_TF}"
            )
            # print_cm(confusion_matrix(y_true, y_pred, labels=range(len(le.classes_))), labels=[l[:] for l in le.classes_.tolist()])
            return (
                (correct_A / total_A, f1_A),
                (correct_B / total_B, f1_B),
                (correct_TF / total_TF, f1_TF),
            )

    def set_past(self, sent, which_task):
        "sent: str, a whole sent"
        # assert sent.startswith("A:") or sent.startswith("B:")
        if sent.startswith("A:") or sent.startswith("B:"):
            pdb.set_trace()
            sent = sent[2:]

        if which_task == "A":
            lm_model = self.model_A
            prefix = "A:"
            device = lm_model.device
        elif which_task == "B":
            lm_model = self.model_B
            prefix = "B:"
            device = lm_model.device
        elif which_task == "TF":
            lm_model = self.model_A
            prefix = "A:"
            # candidate_sent = prefix+" ".join(separate_sents)
            device = lm_model.device

        # encode sent
        self.history.append(prefix + sent)
        sent = (
            self.tokenizer.encode(prefix)
            + self.tokenizer.encode(sent)
            + self.train_dataset.turn_ending
        )
        sent = torch.LongTensor(sent).unsqueeze(0).to(device)

        past = self.move_to_device(self.past, lm_model)
        _, past, _ = lm_model(sent, past)

        self.past = past

    def predict(self, separate_sents, which_task):
        "separate_sents: list of sentences with no prefix"
        past = self.past

        if which_task == "A":
            lm_model = self.model_A
            clf_head = self.clf_A
            le = self.le_A
            prefix = "A:"
            device = lm_model.device
        elif which_task == "B":
            lm_model = self.model_B
            clf_head = self.clf_B
            le = self.le_B
            prefix = "B:"
            device = lm_model.device
        elif which_task == "TF":
            lm_model = self.model_A
            clf_head = self.clf_TF
            prefix = "A:"
            candidate_sent = " ".join(separate_sents)
            device = lm_model.device

        # evaluation mode
        self.model_A.eval()
        self.model_B.eval()
        self.clf_A.eval()
        self.clf_B.eval()
        self.clf_TF.eval()

        with torch.no_grad():
            if which_task in ["A", "B"]:
                all_logits = []
                for i, sent in enumerate(separate_sents):
                    if i == 0:
                        sent = self.tokenizer.encode(prefix) + self.tokenizer.encode(
                            sent
                        )
                    else:
                        sent = self.tokenizer.encode(" " + sent)

                    # pdb.set_trace()
                    sent = torch.LongTensor(sent).unsqueeze(0).to(device)
                    past = self.move_to_device(past, lm_model)
                    logits, past, hidden_states = lm_model(sent, past)

                    # encode [CLS]
                    cls_token_tensor = (
                        torch.LongTensor([self.cls_token_id]).unsqueeze(0).to(device)
                    )
                    _, _, hidden_states = lm_model(cls_token_tensor, past)
                    hidden_states = self.move_to_device(hidden_states, clf_head)
                    mc_logits = clf_head(hidden_states[-1], cls_index=None).squeeze(-1)

                    all_logits.append(mc_logits)

                # finish tail
                end_input = (
                    torch.LongTensor(self.train_dataset.turn_ending)
                    .unsqueeze(0)
                    .to(device)
                )
                past = self.move_to_device(past, lm_model)
                _, past, _ = lm_model(end_input, past)

                # get labels
                all_logits = torch.cat(all_logits, dim=0)
                # pdb.set_trace()
                _, predicted_acts = torch.max(all_logits, 1)
                predicted_acts = predicted_acts.tolist()
                predicted_acts = le.inverse_transform(predicted_acts).tolist()

                return predicted_acts, past
            elif which_task == "TF":
                # encode candidate
                candidate = self.tokenizer.encode(prefix) + self.tokenizer.encode(
                    candidate_sent
                )
                # pdb.set_trace()
                candidate = torch.LongTensor(candidate).unsqueeze(0).to(device)
                past = self.move_to_device(past, self.model_A)
                logits, past, hidden_states = self.model_A(candidate, past)
                # encode [CLS]
                cls_token_tensor = (
                    torch.LongTensor([self.cls_token_id]).unsqueeze(0).to(device)
                )
                _, _, hidden_states = self.model_A(cls_token_tensor, past)
                hidden_states = self.move_to_device(hidden_states, self.clf_TF)
                mc_logits = self.clf_TF(hidden_states[-1], cls_index=None).squeeze(-1)
                # pdb.set_trace()
                _, predicted_acts = torch.max(mc_logits, 1)
                predicted_acts = predicted_acts.tolist()
                assert len(predicted_acts) == 1
                return predicted_acts[0], past

    def train_one_iter(
        self,
        batch,
        batch_TF,
        update_count,
        which_to_train,
        fp16=False,
        is_validation=False,
    ):
        # role_ids, whole_sents, separate_sents, acts = batch
        past = None
        all_sents_A, all_logits_A, all_acts_A = [], [], []
        all_sents_B, all_logits_B, all_acts_B = [], [], []
        for i, (role_id, whole_sent, separate_sents, acts) in enumerate(zip(*batch)):
            if role_id == 0:
                whole_sent = torch.LongTensor(whole_sent).unsqueeze(0).to(self.device1)
                try:
                    assert self.tokenizer.decode(whole_sent[0][:2].tolist()) == "A:"
                except:
                    pdb.set_trace()
                if "A" in which_to_train:
                    past = self.move_to_device(past, self.model_A)
                    _, real_past, _ = self.model_A(whole_sent, past)
                    for act, sent in zip(acts, separate_sents):
                        all_sents_A.append(self.tokenizer.decode(sent))
                        # pdb.set_trace()
                        # 'A:HI I would like to tell you About a childrens charity called Save the CHildren.'
                        sent = torch.LongTensor(sent).unsqueeze(0).to(self.device1)
                        past = self.move_to_device(past, self.model_A)
                        logits, past, hidden_states = self.model_A(sent, past)

                        # pdb.set_trace()
                        # encode [CLS]
                        cls_token_tensor = (
                            torch.LongTensor([self.cls_token_id])
                            .unsqueeze(0)
                            .to(self.device1)
                        )
                        past = self.move_to_device(past, self.model_A)
                        _, _, hidden_states = self.model_A(cls_token_tensor, past)

                        mc_logits = self.clf_A(
                            hidden_states[-1], cls_index=None
                        ).squeeze(-1)
                        all_logits_A.append(mc_logits)
                        all_acts_A.append(act)
                    # pdb.set_trace()
                    past = real_past
                    # # finish tail
                    # end_input = torch.LongTensor(self.train_dataset.turn_ending).unsqueeze(0).to(self.device1)
                    # _, past, _ = self.model_A(end_input, past)
                else:
                    past = self.move_to_device(past, self.model_A)
                    _, past, hidden_states = self.model_A(whole_sent, past)
            else:
                whole_sent = torch.LongTensor(whole_sent).unsqueeze(0).to(self.device2)
                try:
                    assert self.tokenizer.decode(whole_sent[0][:2].tolist()) == "B:"
                except:
                    pdb.set_trace()
                if "B" in which_to_train:
                    past = self.move_to_device(past, self.model_B)
                    _, real_past, _ = self.model_B(whole_sent, past)
                    for act, sent in zip(acts, separate_sents):
                        all_sents_B.append(self.tokenizer.decode(sent))
                        # pdb.set_trace()
                        #'B:ok please do'
                        sent = torch.LongTensor(sent).unsqueeze(0).to(self.device2)
                        past = self.move_to_device(past, self.model_B)
                        logits, past, hidden_states = self.model_B(sent, past)

                        # encode [CLS]
                        cls_token_tensor = (
                            torch.LongTensor([self.cls_token_id])
                            .unsqueeze(0)
                            .to(self.device2)
                        )
                        _, _, hidden_states = self.model_B(cls_token_tensor, past)

                        hidden_states = self.move_to_device(hidden_states, self.clf_B)
                        mc_logits = self.clf_B(
                            hidden_states[-1], cls_index=None
                        ).squeeze(-1)
                        all_logits_B.append(mc_logits)
                        all_acts_B.append(act)
                    past = real_past
                    # finish tail
                    # end_input = torch.LongTensor(self.train_dataset.turn_ending).unsqueeze(0).to(self.device2)
                    # past = self.move_to_device(past, self.model_B)
                    # _, past, _ = self.model_B(end_input, past)
                else:
                    past = self.move_to_device(past, self.model_B)
                    _, past, hidden_states = self.model_B(whole_sent, past)

        all_logits_A = torch.cat(all_logits_A, dim=0)
        all_acts_A = torch.tensor(all_acts_A).unsqueeze(0).to(self.device1)
        # pdb.set_trace()
        loss_A = self.criterion(
            all_logits_A.view(-1, all_logits_A.size(-1)), all_acts_A.view(-1)
        )

        all_logits_B = torch.cat(all_logits_B, dim=0)
        all_acts_B = torch.tensor(all_acts_B).unsqueeze(0).to(self.device1)

        loss_B = self.criterion(
            all_logits_B.view(-1, all_logits_B.size(-1)), all_acts_B.view(-1)
        )

        # TF task
        all_contexts_candidate_TF = []
        all_logits_TF = []
        all_acts_TF = []
        for one_dial in batch_TF:
            past = None
            contexts, candidate, pick_or_not = one_dial
            all_contexts_candidate_TF.append(
                (
                    " ".join([self.tokenizer.decode(c) for c in contexts]),
                    self.tokenizer.decode(candidate),
                )
            )

            # get past
            for i, context in enumerate(contexts):
                if i % 2 == 0:
                    # pdb.set_trace()
                    #'A:Would you like to know more about the charity Save the Children?\n\n\n'
                    context = torch.LongTensor(context).unsqueeze(0).to(self.device1)
                    past = self.move_to_device(past, self.model_A)
                    logits, past, hidden_states = self.model_A(context, past)
                else:
                    # pdb.set_trace()
                    #'B:hello I am great.\n\n\n'
                    context = torch.LongTensor(context).unsqueeze(0).to(self.device2)
                    past = self.move_to_device(past, self.model_B)
                    logits, past, hidden_states = self.model_B(context, past)

            # encode candidate
            # pdb.set_trace()
            # "A:Save the Children is an international non-governmental organization that promotes children's rights, provides relief and helps support children in developing countries."
            candidate = torch.LongTensor(candidate).unsqueeze(0).to(self.device1)
            past = self.move_to_device(past, self.model_A)
            logits, past, hidden_states = self.model_A(candidate, past)
            # encode [CLS]
            cls_token_tensor = (
                torch.LongTensor([self.cls_token_id]).unsqueeze(0).to(self.device1)
            )
            _, _, hidden_states = self.model_A(cls_token_tensor, past)

            mc_logits = self.clf_TF(hidden_states[-1], cls_index=None).squeeze(-1)
            all_logits_TF.append(mc_logits)
            all_acts_TF.append(pick_or_not)

        all_logits_TF = torch.cat(all_logits_TF, dim=0)
        all_acts_TF = torch.tensor(all_acts_TF).unsqueeze(0).to(self.device1)

        loss_TF = self.criterion(
            all_logits_TF.view(-1, all_logits_TF.size(-1)), all_acts_TF.view(-1)
        )

        if is_validation:
            return (
                all_sents_A,
                all_logits_A,
                all_acts_A,
                all_sents_B,
                all_logits_B,
                all_acts_B,
                all_contexts_candidate_TF,
                all_logits_TF,
                all_acts_TF,
            )

        loss = (
            loss_A.to(self.device1) + loss_B.to(self.device1) + loss_TF.to(self.device1)
        )

        loss /= self.num_gradients_accumulation

        if fp16:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        record_loss = loss.item() * self.num_gradients_accumulation

        return record_loss  # , perplexity

    def move_to_device(self, past, target):
        if past is not None and target.device != past[0].device:
            past = [p.to(target.device) for p in past]
        return past


def build_model_classifier(model_dir, device1, device2, models_used_in_model_clf=None):

    config = GPT2Config()
    config = config.from_pretrained("gpt2")  # config.from_pretrained('gpt2-medium')
    config.summary_first_dropout = 0.2
    config.summary_type = "cls_index"

    if models_used_in_model_clf is None:
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # torch.load(tokenizer_dir)
        tokenizer.add_special_tokens({"cls_token": "[CLS]"})
        # device1 = torch.device("cuda:0")
        # device2 = torch.device("cuda:1")
        model_A, model_B = load_model(cfg, "small", tokenizer, device1, device2)

        # pdb.set_trace()
        print("model_clf device\n\n\n\n\n\n")
        print(model_A.device)
        print(model_B.device)
        print("here\n\n\n")
        which_to_train = ["A", "B", "TF"]
        model_clf = ModelClassifier(
            config=config,
            which_to_train=which_to_train,
            model_A=model_A,
            model_B=model_B,
            tokenizer=tokenizer,
            device1=device1,
            device2=device2,
        )

        model_clf.load_model(all_model_dir=model_dir)

        for param in model_clf.parameters():
            param.requires_grad = False

    else:
        tokenizer, model_A, model_B, clf_A, clf_B, clf_TF = models_used_in_model_clf
        # pdb.set_trace()
        print("use predefined models for model_clf!!!\n\n\n\n\n\n")
        print("model_clf device\n\n\n\n\n\n")
        print(model_A.device)
        print(model_B.device)
        print("here\n\n\n")
        which_to_train = ["A", "B", "TF"]
        model_clf = ModelClassifier(
            config=config,
            which_to_train=which_to_train,
            model_A=model_A,
            model_B=model_B,
            tokenizer=tokenizer,
            device1=device1,
            device2=device2,
            clf_A=clf_A,
            clf_B=clf_B,
            clf_TF=clf_TF,
        )

        for param in model_clf.parameters():
            param.requires_grad = False

    return model_clf


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--load", action="store_true", help="load model")
    parser.add_argument("-t", "--test", action="store_true", help="test model")
    parser.add_argument(
        "-v", "--validation", action="store_true", help="validate model"
    )
    parser.add_argument("-n", "--num_epoch", type=int, default=10, help="num of epoch")
    # parser.add_argument('-d', '--device', type=str, default='cpu',
    #                      help='device to use')
    parser.add_argument(
        "-m",
        "--model_dir",
        type=str,
        # default="Checkpoint_act_clf/multitask_TF_best_acc_0.7777777777777778_f1_0.776536312849162_A_acc_0.6954838709677419_f1_0.6707423935799665_B_acc_0.6166134185303515_f1_0.5898033645875225.pth",
        help="model dir",
    )
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = True
    torch.manual_seed(123)
    np.random.seed(123)

    config = GPT2Config()
    config = config.from_pretrained("gpt2")  # config.from_pretrained('gpt2-medium')
    config.summary_first_dropout = 0.2
    config.summary_type = "cls_index"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")  # torch.load(tokenizer_dir)
    tokenizer.add_special_tokens({"cls_token": "[CLS]"})
    device1 = torch.device("cuda:1")
    device2 = torch.device("cuda:0")
    model_A, model_B = load_model(cfg, "small", tokenizer, device1, device2)

    print("device\n\n\n\n\n\n")
    print(model_A.device)
    print(model_B.device)
    print("here\n\n\n")
    which_to_train = ["A", "B", "TF"]
    model_clf = ModelClassifier(
        config=config,
        which_to_train=which_to_train,
        model_A=model_A,
        model_B=model_B,
        tokenizer=tokenizer,
        device1=device1,
        device2=device2,
    )

    # training
    # model_clf.validate(model_clf.val_dataloader, model_clf.val_dataloader_TF, 0, which_to_train)
    if not args.test and not args.validation:
        if args.load:
            model_clf.load_model(all_model_dir=args.model_dir)
        model_clf.train(which_to_train=which_to_train, num_epochs=args.num_epoch)
    elif args.validation:
        model_clf.load_model(all_model_dir=args.model_dir)
        model_clf.validate(
            model_clf.val_dataloader, model_clf.val_dataloader_TF, 0, which_to_train
        )
    elif args.test:
        model_clf.load_model(all_model_dir=args.model_dir)
        # past = None
        while True:
            input_text = input("input: ")
            which_task = input("task: ")
            input_texts = sent_tokenize(input_text)
            predicted_acts, _ = model_clf.predict(
                separate_sents=input_texts, which_task=which_task
            )
            print(predicted_acts)
