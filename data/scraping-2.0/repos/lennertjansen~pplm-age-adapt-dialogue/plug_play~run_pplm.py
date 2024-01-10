#! /usr/bin/env python3
# coding=utf-8
# Copyright 2018 The Uber AI Team Authors.
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

"""
Example command with bag of words:
python examples/run_pplm.py -B space --cond_text "The president" --length 100 --gamma 1.5 --num_iterations 3 --num_samples 10 --stepsize 0.01 --window_length 5 --kl_scale 0.01 --gm_scale 0.95

Example command with discriminator:
python examples/run_pplm.py -D sentiment --class_label 3 --cond_text "The lake" --length 10 --gamma 1.0 --num_iterations 30 --num_samples 10 --stepsize 0.01 --kl_scale 0.01 --gm_scale 0.95
"""

import argparse
import json
import pdb
from operator import add
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import trange
from transformers import GPT2Tokenizer
from transformers.file_utils import cached_path
# from transformers.modeling_gpt2 import GPT2LMHeadModel

from pplm_classification_head import ClassificationHead

##### Imports added by Lennert
from nltk import ngrams # for distinct ngrams function
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelWithLMHead, AutoModelForCausalLM

from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
import pandas as pd
from pdb import set_trace
import re
from nltk.corpus import stopwords
from datetime import datetime

# LJ: Imports for BERT
from transformers import BertTokenizer

from torch.utils.data import DataLoader

# LJ: doing this so I can import from the parent directory. Found this trick here: https://www.delftstack.com/howto/python/python-import-from-parent-directory/
import os, sys
p = os.path.abspath('.')
sys.path.insert(1, p)
from classifiers import TextClassificationBERT
from dataset import BncDataset, PadSequence
from transformers import BertTokenizer, BertModel

PPLM_BOW = 1
PPLM_DISCRIM = 2
PPLM_BOW_DISCRIM = 3
SMALL_CONST = 1e-15
BIG_CONST = 1e10

QUIET = 0
REGULAR = 1
VERBOSE = 2
VERY_VERBOSE = 3
VERBOSITY_LEVELS = {
    'quiet': QUIET,
    'regular': REGULAR,
    'verbose': VERBOSE,
    'very_verbose': VERY_VERBOSE,
}

BAG_OF_WORDS_ARCHIVE_MAP = {
    'legal': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/legal.txt",
    'military': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/military.txt",
    'monsters': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/monsters.txt",
    'politics': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/politics.txt",
    'positive_words': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/positive_words.txt",
    'religion': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/religion.txt",
    'science': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/science.txt",
    'space': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/space.txt",
    'technology': "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/bow/technology.txt",
}

DISCRIMINATOR_MODELS_PARAMS = {
    "clickbait": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/clickbait_classifier_head.pt",
        "class_size": 2,
        "embed_size": 1024,
        "class_vocab": {"non_clickbait": 0, "clickbait": 1},
        "default_class": 1,
        "pretrained_model": "gpt2-medium",
    },
    "sentiment": {
        "url": "https://s3.amazonaws.com/models.huggingface.co/bert/pplm/discriminators/SST_classifier_head.pt",
        "class_size": 5,
        "embed_size": 1024,
        "class_vocab": {"very_positive": 2, "very_negative": 3},
        "default_class": 3,
        "pretrained_model": "gpt2-medium",
    },
}


def to_var(x, requires_grad=False, volatile=False, device='cuda'):
    if torch.cuda.is_available() and device == 'cuda':
        x = x.cuda()
    elif device != 'cuda':
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad, volatile=volatile)


def top_k_filter(logits, k, probs=False):
    """
    Masks everything but the k top entries as -infinity (1e10).
    Used to mask logits such that e^-infinity -> 0 won't contribute to the
    sum of the denominator.
    """
    if k == 0:
        return logits
    else:
        values = torch.topk(logits, k)[0]
        batch_mins = values[:, -1].view(-1, 1).expand_as(logits)
        if probs:
            return torch.where(logits < batch_mins,
                               torch.ones_like(logits) * 0.0, logits)
        return torch.where(logits < batch_mins,
                           torch.ones_like(logits) * -BIG_CONST,
                           logits)


def perturb_past(
        past,
        model,
        last,
        unpert_past=None,
        unpert_logits=None,
        accumulated_hidden=None,
        grad_norms=None,
        stepsize=0.01,
        one_hot_bows_vectors=None,
        classifier=None,
        class_label=None,
        loss_type=0,
        num_iterations=3,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        kl_scale=0.01,
        device='cuda',
        verbosity_level=REGULAR
):
    # Generate inital perturbed past
    grad_accumulator = [
        (np.zeros(p.shape).astype("float32"))
        for p in past
    ]

    if accumulated_hidden is None:
        accumulated_hidden = 0

    if decay:
        decay_mask = torch.arange(
            0.,
            1.0 + SMALL_CONST,
            1.0 / (window_length)
        )[1:]
    else:
        decay_mask = 1.0

    # TODO fix this comment (SUMANTH)
    # Generate a mask is gradient perturbated is based on a past window
    _, _, _, curr_length, _ = past[0].shape

    if curr_length > window_length and window_length > 0:
        ones_key_val_shape = (
                tuple(past[0].shape[:-2])
                + tuple([window_length])
                + tuple(past[0].shape[-1:])
        )

        zeros_key_val_shape = (
                tuple(past[0].shape[:-2])
                + tuple([curr_length - window_length])
                + tuple(past[0].shape[-1:])
        )

        ones_mask = torch.ones(ones_key_val_shape)
        ones_mask = decay_mask * ones_mask.permute(0, 1, 2, 4, 3)
        ones_mask = ones_mask.permute(0, 1, 2, 4, 3)

        window_mask = torch.cat(
            (ones_mask, torch.zeros(zeros_key_val_shape)),
            dim=-2
        ).to(device)
    else:
        window_mask = torch.ones_like(past[0]).to(device)

    # accumulate perturbations for num_iterations
    loss_per_iter = []
    new_accumulated_hidden = None
    for i in range(num_iterations):
        if verbosity_level >= VERBOSE:
            print("Iteration ", i + 1)
        curr_perturbation = [
            to_var(torch.from_numpy(p_), requires_grad=True, device=device)
            for p_ in grad_accumulator
        ]

        # Compute hidden using perturbed past
        perturbed_past = list(map(add, past, curr_perturbation))
        _, _, _, curr_length, _ = curr_perturbation[0].shape
        all_logits, _, all_hidden = model(last, past=perturbed_past)
        hidden = all_hidden[-1]
        new_accumulated_hidden = accumulated_hidden + torch.sum(
            hidden,
            dim=1
        ).detach()
        # TODO: Check the layer-norm consistency of this with trained discriminator (Sumanth)
        logits = all_logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)

        loss = 0.0
        loss_list = []
        if loss_type == PPLM_BOW or loss_type == PPLM_BOW_DISCRIM:
            for one_hot_bow in one_hot_bows_vectors:
                bow_logits = torch.mm(probs, torch.t(one_hot_bow))
                bow_loss = -torch.log(torch.sum(bow_logits))
                loss += bow_loss
                loss_list.append(bow_loss)
            if verbosity_level >= VERY_VERBOSE:
                print(" pplm_bow_loss:", loss.data.cpu().numpy())

        if loss_type == PPLM_DISCRIM or loss_type == PPLM_BOW_DISCRIM:
            ce_loss = torch.nn.CrossEntropyLoss()
            # TODO why we need to do this assignment and not just using unpert_past? (Sumanth)
            curr_unpert_past = unpert_past
            curr_probs = torch.unsqueeze(probs, dim=1)
            wte = model.resize_token_embeddings()
            for _ in range(horizon_length):
                inputs_embeds = torch.matmul(curr_probs, wte.weight.data)
                _, curr_unpert_past, curr_all_hidden = model(
                    past=curr_unpert_past,
                    inputs_embeds=inputs_embeds
                )
                curr_hidden = curr_all_hidden[-1]
                new_accumulated_hidden = new_accumulated_hidden + torch.sum(
                    curr_hidden, dim=1)

            prediction = classifier(new_accumulated_hidden /
                                    (curr_length + 1 + horizon_length))

            label = torch.tensor(prediction.shape[0] * [class_label],
                                 device=device,
                                 dtype=torch.long)
            discrim_loss = ce_loss(prediction, label)
            if verbosity_level >= VERY_VERBOSE:
                print(" pplm_discrim_loss:", discrim_loss.data.cpu().numpy())
            loss += discrim_loss
            loss_list.append(discrim_loss)

        kl_loss = 0.0
        if kl_scale > 0.0:
            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)
            unpert_probs = (
                    unpert_probs + SMALL_CONST *
                    (unpert_probs <= SMALL_CONST).float().to(device).detach()
            )
            correction = SMALL_CONST * (probs <= SMALL_CONST).float().to(
                device).detach()
            corrected_probs = probs + correction.detach()
            kl_loss = kl_scale * (
                (corrected_probs * (corrected_probs / unpert_probs).log()).sum()
            )
            if verbosity_level >= VERY_VERBOSE:
                print(' kl_loss', kl_loss.data.cpu().numpy())
            loss += kl_loss

        loss_per_iter.append(loss.data.cpu().numpy())
        if verbosity_level >= VERBOSE:
            print(' pplm_loss', (loss - kl_loss).data.cpu().numpy())

        # compute gradients
        loss.backward()

        # calculate gradient norms
        if grad_norms is not None and loss_type == PPLM_BOW:
            grad_norms = [
                torch.max(grad_norms[index], torch.norm(p_.grad * window_mask))
                for index, p_ in enumerate(curr_perturbation)
            ]
        else:
            grad_norms = [
                (torch.norm(p_.grad * window_mask) + SMALL_CONST)
                for index, p_ in enumerate(curr_perturbation)
            ]

        # normalize gradients
        grad = [
            -stepsize *
            (p_.grad * window_mask / grad_norms[
                index] ** gamma).data.cpu().numpy()
            for index, p_ in enumerate(curr_perturbation)
        ]

        # accumulate gradient
        grad_accumulator = list(map(add, grad, grad_accumulator))

        # reset gradients, just to make sure
        for p_ in curr_perturbation:
            p_.grad.data.zero_()

        # removing past from the graph
        new_past = []
        for p_ in past:
            new_past.append(p_.detach())
        past = new_past

    # apply the accumulated perturbations to the past
    grad_accumulator = [
        to_var(torch.from_numpy(p_), requires_grad=True, device=device)
        for p_ in grad_accumulator
    ]
    pert_past = list(map(add, past, grad_accumulator))

    return pert_past, new_accumulated_hidden, grad_norms, loss_per_iter


def get_classifier(
        name: Optional[str],
        class_label: Union[str, int],
        device: str,
        verbosity_level: int = REGULAR
) -> Tuple[Optional[ClassificationHead], Optional[int]]:
    if name is None:
        return None, None

    params = DISCRIMINATOR_MODELS_PARAMS[name]
    classifier = ClassificationHead(
        class_size=params['class_size'],
        embed_size=params['embed_size']
    ).to(device)
    if "url" in params:
        resolved_archive_file = cached_path(params["url"])
    elif "path" in params:
        resolved_archive_file = params["path"]
    else:
        raise ValueError("Either url or path have to be specified "
                         "in the discriminator model parameters")
    classifier.load_state_dict(
        torch.load(resolved_archive_file, map_location=device))
    classifier.eval()

    if isinstance(class_label, str):
        if class_label in params["class_vocab"]:
            label_id = params["class_vocab"][class_label]
        else:
            label_id = params["default_class"]
            if verbosity_level >= REGULAR:
                print("class_label {} not in class_vocab".format(class_label))
                print("available values are: {}".format(params["class_vocab"]))
                print("using default class {}".format(label_id))

    elif isinstance(class_label, int):
        if class_label in set(params["class_vocab"].values()):
            label_id = class_label
        else:
            label_id = params["default_class"]
            if verbosity_level >= REGULAR:
                print("class_label {} not in class_vocab".format(class_label))
                print("available values are: {}".format(params["class_vocab"]))
                print("using default class {}".format(label_id))

    else:
        label_id = params["default_class"]

    return classifier, label_id


def get_bag_of_words_indices(bag_of_words_ids_or_paths: List[str], tokenizer) -> \
        List[List[List[int]]]:
    bow_indices = []
    for id_or_path in bag_of_words_ids_or_paths:
        if id_or_path in BAG_OF_WORDS_ARCHIVE_MAP:
            filepath = cached_path(BAG_OF_WORDS_ARCHIVE_MAP[id_or_path])
        else:
            filepath = id_or_path
        with open(filepath, "r") as f:
            words = f.read().strip().split("\n")
        bow_indices.append(
            [tokenizer.encode(word.strip(),
                              add_prefix_space=True,
                              add_special_tokens=False)
             for word in words])
    return bow_indices


def build_bows_one_hot_vectors(bow_indices, tokenizer, device='cuda'):
    if bow_indices is None:
        return None

    one_hot_bows_vectors = []
    for single_bow in bow_indices:
        single_bow = list(filter(lambda x: len(x) <= 1, single_bow))
        single_bow = torch.tensor(single_bow).to(device)
        num_words = single_bow.shape[0]
        one_hot_bow = torch.zeros(num_words, tokenizer.vocab_size).to(device)
        one_hot_bow.scatter_(1, single_bow, 1)
        one_hot_bows_vectors.append(one_hot_bow)
    return one_hot_bows_vectors


def full_text_generation(
        model,
        tokenizer,
        context=None,
        num_samples=1,
        device="cuda",
        bag_of_words=None,
        discrim=None,
        class_label=None,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        verbosity_level=REGULAR,
        **kwargs
):
    classifier, class_id = get_classifier(
        discrim,
        class_label,
        device
    )

    bow_indices = []
    if bag_of_words:
        bow_indices = get_bag_of_words_indices(bag_of_words.split(";"),
                                               tokenizer)

    if bag_of_words and classifier:
        loss_type = PPLM_BOW_DISCRIM
        if verbosity_level >= REGULAR:
            print("Both PPLM-BoW and PPLM-Discrim are on. "
                  "This is not optimized.")

    elif bag_of_words:
        loss_type = PPLM_BOW
        if verbosity_level >= REGULAR:
            print("Using PPLM-BoW")

    elif classifier is not None:
        loss_type = PPLM_DISCRIM
        if verbosity_level >= REGULAR:
            print("Using PPLM-Discrim")

    else:
        raise Exception("Specify either a bag of words or a discriminator")

    unpert_gen_tok_text, _, _ = generate_text_pplm(
        model=model,
        tokenizer=tokenizer,
        context=context,
        device=device,
        length=length,
        sample=sample,
        perturb=False,
        verbosity_level=verbosity_level
    )
    if device == 'cuda':
        torch.cuda.empty_cache()

    pert_gen_tok_texts = []
    discrim_losses = []
    losses_in_time = []

    for i in range(num_samples):
        pert_gen_tok_text, discrim_loss, loss_in_time = generate_text_pplm(
            model=model,
            tokenizer=tokenizer,
            context=context,
            device=device,
            perturb=True,
            bow_indices=bow_indices,
            classifier=classifier,
            class_label=class_id,
            loss_type=loss_type,
            length=length,
            stepsize=stepsize,
            temperature=temperature,
            top_k=top_k,
            sample=sample,
            num_iterations=num_iterations,
            grad_length=grad_length,
            horizon_length=horizon_length,
            window_length=window_length,
            decay=decay,
            gamma=gamma,
            gm_scale=gm_scale,
            kl_scale=kl_scale,
            verbosity_level=verbosity_level
        )
        pert_gen_tok_texts.append(pert_gen_tok_text)
        if classifier is not None:
            discrim_losses.append(discrim_loss.data.cpu().numpy())
        losses_in_time.append(loss_in_time)

    if device == 'cuda':
        torch.cuda.empty_cache()

    return unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time


def generate_text_pplm(
        model,
        tokenizer,
        context=None,
        past=None,
        device="cuda",
        perturb=True,
        bow_indices=None,
        classifier=None,
        class_label=None,
        loss_type=0,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        verbosity_level=REGULAR
):
    output_so_far = None
    if context:
        context_t = torch.tensor(context, device=device, dtype=torch.long)
        while len(context_t.shape) < 2:
            context_t = context_t.unsqueeze(0)
        output_so_far = context_t

    # collect one hot vectors for bags of words
    one_hot_bows_vectors = build_bows_one_hot_vectors(bow_indices, tokenizer,
                                                      device)

    grad_norms = None
    last = None
    unpert_discrim_loss = 0
    loss_in_time = []

    if verbosity_level >= VERBOSE:
        range_func = trange(length, ascii=True)
    else:
        range_func = range(length)

    for i in range_func:

        # Get past/probs for current output, except for last word
        # Note that GPT takes 2 inputs: past + current_token

        # run model forward to obtain unperturbed
        if past is None and output_so_far is not None:
            last = output_so_far[:, -1:]
            if output_so_far.shape[1] > 1:
                _, past, _ = model(output_so_far[:, :-1])

        unpert_logits, unpert_past, unpert_all_hidden = model(output_so_far)
        unpert_last_hidden = unpert_all_hidden[-1]

        # check if we are abowe grad max length
        if i >= grad_length:
            current_stepsize = stepsize * 0
        else:
            current_stepsize = stepsize

        # modify the past if necessary
        if not perturb or num_iterations == 0:
            pert_past = past

        else:
            accumulated_hidden = unpert_last_hidden[:, :-1, :]
            accumulated_hidden = torch.sum(accumulated_hidden, dim=1)

            if past is not None:
                pert_past, _, grad_norms, loss_this_iter = perturb_past(
                    past,
                    model,
                    last,
                    unpert_past=unpert_past,
                    unpert_logits=unpert_logits,
                    accumulated_hidden=accumulated_hidden,
                    grad_norms=grad_norms,
                    stepsize=current_stepsize,
                    one_hot_bows_vectors=one_hot_bows_vectors,
                    classifier=classifier,
                    class_label=class_label,
                    loss_type=loss_type,
                    num_iterations=num_iterations,
                    horizon_length=horizon_length,
                    window_length=window_length,
                    decay=decay,
                    gamma=gamma,
                    kl_scale=kl_scale,
                    device=device,
                    verbosity_level=verbosity_level
                )
                loss_in_time.append(loss_this_iter)
            else:
                pert_past = past

        pert_logits, past, pert_all_hidden = model(last, past_key_values=pert_past) # i renamed past --> past_key_values
        pert_logits = pert_logits[:, -1, :] / temperature  # + SMALL_CONST
        pert_probs = F.softmax(pert_logits, dim=-1)

        if classifier is not None:
            ce_loss = torch.nn.CrossEntropyLoss()
            prediction = classifier(torch.mean(unpert_last_hidden, dim=1))
            label = torch.tensor([class_label], device=device,
                                 dtype=torch.long)
            unpert_discrim_loss = ce_loss(prediction, label)
            if verbosity_level >= VERBOSE:
                print(
                    "unperturbed discrim loss",
                    unpert_discrim_loss.data.cpu().numpy()
                )
        else:
            unpert_discrim_loss = 0

        # Fuse the modified model and original model
        if perturb:

            unpert_probs = F.softmax(unpert_logits[:, -1, :], dim=-1)

            pert_probs = ((pert_probs ** gm_scale) * (
                    unpert_probs ** (1 - gm_scale)))  # + SMALL_CONST
            pert_probs = top_k_filter(pert_probs, k=top_k,
                                      probs=True)  # + SMALL_CONST

            # rescale
            if torch.sum(pert_probs) <= 1:
                pert_probs = pert_probs / torch.sum(pert_probs)

        else:
            pert_logits = top_k_filter(pert_logits, k=top_k)  # + SMALL_CONST
            pert_probs = F.softmax(pert_logits, dim=-1)

        # sample or greedy
        if sample:
            last = torch.multinomial(pert_probs, num_samples=1)

        else:
            _, last = torch.topk(pert_probs, k=1, dim=-1)

        # update context/output_so_far appending the new token
        output_so_far = (
            last if output_so_far is None
            else torch.cat((output_so_far, last), dim=1)
        )
        if verbosity_level >= REGULAR:
            print(tokenizer.decode(output_so_far.tolist()[0]))

    return output_so_far, unpert_discrim_loss, loss_in_time


def set_generic_model_params(discrim_weights, discrim_meta):
    if discrim_weights is None:
        raise ValueError('When using a generic discriminator, '
                         'discrim_weights need to be specified')
    if discrim_meta is None:
        raise ValueError('When using a generic discriminator, '
                         'discrim_meta need to be specified')

    with open(discrim_meta, 'r') as discrim_meta_file:
        meta = json.load(discrim_meta_file)
    meta['path'] = discrim_weights
    DISCRIMINATOR_MODELS_PARAMS['generic'] = meta

# LJ:
# Add labels for age categories
def age_to_cat(label):
    '''Returns age category label for given age number.'''

    if label == '19_29':
        return 0  # '13-17'
    elif label == '50_plus':
        return 1  # '23-27'
    else:
        raise ValueError("Given age not in one of pre-defined age groups.")


# LJ:
def preprocess_col(df):

    data_size = len(df)

    # Remove all non-alphabetical characters
    df['clean_text'] = df['text'].apply(lambda x: re.sub(r'[^A-Za-z]+',' ', x))

    # make all letters lowercase
    df['clean_text'] = df['clean_text'].apply(lambda x: x.lower())

    # remove whitespaces from beginning or ending
    df['clean_text'] = df['clean_text'].apply(lambda x: x.strip())

    # remove stop words
    # stopwords_dict = set(stopwords.words('english')) # use set (hash table) data structure for faster lookup
    # df['clean_text'] = df['clean_text'].apply(lambda x: ' '.join([words for words in x.split() if words not in stopwords_dict]))

    # Remove instances empty strings
    df.drop(df[df.clean_text == ''].index, inplace = True)

    # number of datapoints removed by all pre-processing steps
    dropped_instances = data_size - len(df)
    data_size = len(df)

    # df['age_cat'] = df['label'].apply(age_to_cat)

    # # rename column
    # df.rename(columns={'label': 'age_cat'}, inplace=True)

    return df

# LJ
def perplexity(text,
               device,
               model_id='openai-gpt',
               stride=512):

    # Based on: https://huggingface.co/transformers/perplexity.html
    # model_id = 'gpt2-medium'
    # model = GPT2LMHeadModel.from_pretrained(model_id).to(device)
    # tokenizer = GPT2TokenizerFast.from_pretrained(model_id)

    # model_id = 'openai-gpt'
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # tokenizer = OpenAIGPTTokenizer.from_pretrained(model_id)
    # model = OpenAIGPTLMHeadModel.from_pretrained(model_id).to(device)

    # encode the input text
    encodings = tokenizer('\n\n'.join(text), return_tensors='pt')

    max_length = model.config.n_positions
    # stride = 512

    lls = [] # placeholder for log-likelihoods

    for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings.input_ids.size(1))
        trg_len = end_loc - i  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * trg_len

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / end_loc)

    return ppl.item()

# LJ
def dist_n_grams(text, n):
    # computes the distinct n-grams in a text passage, normalized by passage length
    # ## CHeck this out: https://stackoverflow.com/questions/17531684/n-grams-in-python-four-five-six-grams
    # Possible steps:
    # split text into all n-grams
    # create set() of n-grams to keep distinct ones
    # count set length
    # divide by text length (TODO: is length measured in no. 1-grams always or no. n-grams) --> Li et al 2015 say number of generated words.

    split_text = text.split()
    unique_ngrams = set(list(ngrams(split_text, n)))

    return len(unique_ngrams) / len(split_text)


# LJ
def bert_pred(model, criterion, device, data_loader, data='bnc_rb', writer=None, global_iteration=0, set='validation',
                         print_metrics=True, plot_cm=False, save_fig=True, show_fig=False, model_type='bert', mode='train'):
    # For Confucius matrix
    y_pred = []
    y_true = []
    probabilities = []

    # set model to evaluation mode
    model.eval()

    # initialize loss and number of correct predictions
    set_loss = 0
    total_correct = 0

    # start eval timer
    eval_start_time = datetime.now()

    with torch.no_grad():
        for iteration, (batch_inputs, batch_labels, batch_lengths) in enumerate(data_loader):

            # move everything to device
            batch_inputs, batch_labels, batch_lengths = batch_inputs.to(device), batch_labels.to(device), \
                                                        batch_lengths.to(device)


            loss, text_fea = model(batch_inputs, batch_labels)
            batch_probs = torch.softmax(text_fea, axis=1)
            set_loss += loss

            predictions = torch.argmax(text_fea, 1)


            # batch_pred = [int(item[0]) for item in predictions.tolist()]
            # batch_pred = predictions.tolist()
            # ## OLD
            # if model_type == 'lstm':
            #     y_pred.extend(batch_pred)
            # elif model_type == 'bert':
            #     y_pred.extend(predictions.tolist())

            y_pred.extend(predictions.tolist()) #New
            y_true.extend(batch_labels.tolist())
            probabilities.extend(batch_probs.tolist())

            total_correct += predictions.eq(batch_labels.view_as(predictions)).sum().item()

        # average losses and accuracy
        set_loss /= len(data_loader.dataset)
        accuracy = total_correct / len(data_loader.dataset)
        if print_metrics:
            print('-' * 91)
            print(
                "| " + set + " set "
                "| time {}"
                "| loss: {:.5f} | Accuracy: {}/{} ({:.5f})".format(
                    datetime.now() - eval_start_time, set_loss, total_correct, len(data_loader.dataset), accuracy
                )
            )
            print('-' * 91)

        if writer:
            if set == 'validation':
                writer.add_scalar('Accuracy/val', accuracy, global_iteration)
                writer.add_scalar('Loss/val', set_loss, global_iteration)

        print(91 * '-')
        # print(34 * '-' + ' Classification Report ' + 34 * '-')
        # labels = [label for label in range(data_loader.dataset.num_classes)]
        # print(classification_report(y_true, y_pred, labels=labels, digits=5, zero_division=0))
        #
        # print(91 * '-')
        # print('| Confusion Matrix |')
        # # cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='all')
        # cm = confusion_matrix(y_true, y_pred, labels=labels)
        #
        # # df_confusion = pd.DataFrame(cm * len(y_true))
        # df_confusion = pd.DataFrame(cm)
        # print("    Predicted")
        # print(df_confusion)
        # print("True -->")

        # print(cm * len(y_true))

        # if plot_cm:
        #
        #     if data == 'bnc' or 'bnc_rb':
        #         tick_labels = ['19_29', '50_plus']
        #     elif data == 'blog':
        #         tick_labels = ['13-17', '23-27', '33-47']
        #     make_confusion_matrix(cf=cm, categories=tick_labels, title=f'Confusion Matrix for {data} on {set} set',
        #                           num_labels=labels, y_true=y_true, y_pred=y_pred, figsize=FIGSIZE)
        #
        #     if save_fig:
        #         cur_datetime = datetime.now().strftime('%d_%b_%Y_%H_%M_%S')
        #         plt.savefig(f"{FIGDIR}{data}/cm_{model_type}_{set}_dt_{cur_datetime}.png",
        #                     bbox_inches='tight')
        #     if show_fig:
        #         plt.show()


        if mode == 'tvt':
            f1_scores = f1_score(y_true, y_pred, average=None)

            return set_loss, accuracy, f1_scores
        else:
            return probabilities

#TODO:
# LJ
def eval_text():
    # takes in (single?) text (str) as argument.
    # returns (1) perplexity wrt gpt (?), (2) diversity of text passages, (3) age classification score

    pass

def run_pplm_example(
        pretrained_model="gpt2-medium",
        cond_text="",
        uncond=False,
        num_samples=1,
        bag_of_words=None,
        discrim=None,
        discrim_weights=None,
        discrim_meta=None,
        class_label=-1,
        length=100,
        stepsize=0.02,
        temperature=1.0,
        top_k=10,
        sample=True,
        num_iterations=3,
        grad_length=10000,
        horizon_length=1,
        window_length=0,
        decay=False,
        gamma=1.5,
        gm_scale=0.9,
        kl_scale=0.01,
        seed=0,
        no_cuda=False,
        colorama=False,
        verbosity='regular',
        prompt_type='unknown_prompt'
):
    # set Random seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # LJ
    start_lj = datetime.now()

    # LJ
    if uncond:
        prompt_type = 'unprompted'

    # set verbosiry
    verbosity_level = VERBOSITY_LEVELS.get(verbosity.lower(), REGULAR)

    # set the device
    device = "cuda" if torch.cuda.is_available() and not no_cuda else "cpu"

    if discrim == 'generic':
        set_generic_model_params(discrim_weights, discrim_meta)

    if discrim is not None:
        discriminator_pretrained_model = DISCRIMINATOR_MODELS_PARAMS[discrim][
            "pretrained_model"
        ]
        if pretrained_model != discriminator_pretrained_model:
            pretrained_model = discriminator_pretrained_model
            if verbosity_level >= REGULAR:
                print("discrim = {}, pretrained_model set "
                "to discriminator's = {}".format(discrim, pretrained_model))

    # load pretrained model
    # model = GPT2LMHeadModel.from_pretrained(
    #     pretrained_model,
    #     output_hidden_states=True,
    #     return_dict=False
    # ) # LJ: added "return_dict=False" to solve this error: "AttributeError: 'str' object has no attribute 'size'" based on this thread: https://github.com/allanj/pytorch_neural_crf/issues/22
    # model.to(device)
    # model.eval()
    #
    # # load tokenizer
    # tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)

    # LJ: Use this if-else statement so you can actually choose your pretrained model via commandline args
    if pretrained_model.startswith("gpt2"):
        # load pretrained model and corresponding tokenizer
        model = GPT2LMHeadModel.from_pretrained(
            pretrained_model,
            output_hidden_states=True,
            return_dict=False
        )  # LJ: added "return_dict=False" to solve this error: "AttributeError: 'str' object has no attribute 'size'" based on this thread: https://github.com/allanj/pytorch_neural_crf/issues/22
        tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model)

    elif pretrained_model.startswith("microsoft/DialoGPT"):

        # model = AutoModelForCausalLM.from_pretrained(pretrained_model,
        #                                              output_hidden_states=True,
        #                                              return_dict=False
        #                                              )
        model = AutoModelWithLMHead.from_pretrained(pretrained_model,
                                                    output_hidden_states=True,
                                                    return_dict=False
                                                    )
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    elif pretrained_model.startswith("bert"):
        model = BertModel.from_pretrained(
            pretrained_model,
            output_hidden_states=True,
            return_dict=False
        )  # LJ: added "return_dict=False" to solve this error: "AttributeError: 'str' object has no attribute 'size'" based on this thread: https://github.com/allanj/pytorch_neural_crf/issues/22
        tokenizer = BertTokenizer.from_pretrained(pretrained_model)

    # LJ: move model to device and set to evaluation mode
    model.to(device)
    model.eval()

    # Freeze GPT-2 weights
    for param in model.parameters():
        param.requires_grad = False

    # figure out conditioning text
    if uncond:
        tokenized_cond_text = tokenizer.encode(
            [tokenizer.bos_token],
            add_special_tokens=False
        )
    else:
        raw_text = cond_text
        while not raw_text:
            print("Did you forget to add `--cond_text`? ")
            raw_text = input("Model prompt >>> ")
        tokenized_cond_text = tokenizer.encode(
            tokenizer.bos_token + raw_text,
            add_special_tokens=False
        )

    print("= Prefix of sentence =")
    print(tokenizer.decode(tokenized_cond_text))
    print()

    # generate unperturbed and perturbed texts

    # full_text_generation returns:
    # unpert_gen_tok_text, pert_gen_tok_texts, discrim_losses, losses_in_time
    # start = datetime.now() # LJ
    unpert_gen_tok_text, pert_gen_tok_texts, _, _ = full_text_generation(
        model=model,
        tokenizer=tokenizer,
        context=tokenized_cond_text,
        device=device,
        num_samples=num_samples,
        bag_of_words=bag_of_words,
        discrim=discrim,
        class_label=class_label,
        length=length,
        stepsize=stepsize,
        temperature=temperature,
        top_k=top_k,
        sample=sample,
        num_iterations=num_iterations,
        grad_length=grad_length,
        horizon_length=horizon_length,
        window_length=window_length,
        decay=decay,
        gamma=gamma,
        gm_scale=gm_scale,
        kl_scale=kl_scale,
        verbosity_level=verbosity_level
    )

    # untokenize unperturbed text
    unpert_gen_text = tokenizer.decode(unpert_gen_tok_text.tolist()[0])

    if verbosity_level >= REGULAR:
        print("=" * 80)
    print("= Unperturbed generated text =")
    print(unpert_gen_text)
    # with open('plug_play/texts/tryout.txt', 'a', encoding='utf-8') as f:
    #     f.write("%s\n" % unpert_gen_text[13:])
    end_unpert = datetime.now() #LJ
    # print(f"Time to generate, detokenize, and print unpert. text: {end_unpert - start}") # LJ
    print()

    generated_texts = []

    # LJ: placeholder dataframe for generated texts to be evaluated
    if uncond:
        cond_text = tokenizer.bos_token

    gen_text_df = pd.DataFrame(columns=['label', 'prompt', 'text'])

    bow_word_ids = set()
    if bag_of_words and colorama:
        bow_indices = get_bag_of_words_indices(bag_of_words.split(";"),
                                               tokenizer)
        for single_bow_list in bow_indices:
            # filtering all words in the list composed of more than 1 token
            filtered = list(filter(lambda x: len(x) <= 1, single_bow_list))
            # w[0] because we are sure w has only 1 item because previous fitler
            bow_word_ids.update(w[0] for w in filtered)

    # iterate through the perturbed texts
    for i, pert_gen_tok_text in enumerate(pert_gen_tok_texts):
        try:
            # untokenize unperturbed text
            if colorama:
                import colorama

                pert_gen_text = ''
                for word_id in pert_gen_tok_text.tolist()[0]:
                    if word_id in bow_word_ids:
                        pert_gen_text += '{}{}{}'.format(
                            colorama.Fore.RED,
                            tokenizer.decode([word_id]),
                            colorama.Style.RESET_ALL
                        )
                    else:
                        pert_gen_text += tokenizer.decode([word_id])
            else:
                pert_gen_text = tokenizer.decode(pert_gen_tok_text.tolist()[0])

            print("= Perturbed generated text {} =".format(i + 1))
            print(pert_gen_text)
            # end_pert = datetime.now() # LJ
            # with open('plug_play/texts/tryout.txt', 'a', encoding='utf-8') as f:
            #     f.write("%s\n" % pert_gen_text[13:])
            print()
        except:
            pass

        # print(f"Time to generate, detokenize, and print pert. text: {end_pert - start}") # LJ
        # keep the prefix, perturbed seq, original seq for each index
        generated_texts.append(
            (tokenized_cond_text, pert_gen_tok_text, unpert_gen_tok_text)
        )

        # LJ: append generated text and label to df
        gen_text_df.loc[0 if pd.isnull(gen_text_df.index.max()) else gen_text_df.index.max() + 1] = [class_label] + \
                                                                                                    [cond_text] + \
                                                                                                    [pert_gen_text[len(tokenizer.bos_token) + len(cond_text):].replace("\n","")]



    sys.exit(0) #TODO: Remove this when you're done generating scripted dialogues
    # LJ: add column for text length
    gen_text_df['text_length'] = gen_text_df['text'].apply(lambda x: len(x.split()))



    #######################
    # LJ: Time to test BERT
    # LJ: loss criterion
    criterion = torch.nn.CrossEntropyLoss()  # combines LogSoftmax and NLL

    # LJ: Initialize, move to device, and load saved model
    bert_model = TextClassificationBERT(num_classes=2)
    bert_model.to(device)
    # bert_model_path = 'bert_bnc_rb_case_analysis_seed_4_BEST.pt' # TODO: CHANGE TO BERT TRAINED WITH STOPWORDS!!!!!
    bert_model_path = 'bert_bnc_rb_ws_ca_seed_4_24_Sep_2021_BEST.pt'
    bert_model.load_state_dict(torch.load(bert_model_path, map_location=device))

    # LJ: Setup data stuff
    # LJ: BERT tokenizer
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', truncation=True,
                                              max_length=500)  # truncation only considers sequences of max 512 tokens (same as original BERT implementation)

    # LJ: preprocess sequences for bert
    bert_df = preprocess_col(gen_text_df)[['clean_text', 'label']]

    # LJ: re-rename column because im stupid
    bert_df.rename(columns={'label': 'age_cat'}, inplace=True)

    # LJ: create BncDataset instance
    bert_dataset = BncDataset(df=bert_df, tokenizer=bert_tokenizer)

    #LJ: construct dataloader
    bert_loader = DataLoader(dataset=bert_dataset,
                             batch_size=4,
                             shuffle=False,
                             collate_fn=PadSequence())

    # LJ: use bert to make predictions and save assigned probabilities of belonging to young or old age group
    bert_probs = bert_pred(model=bert_model, criterion=criterion, device=device, data_loader=bert_loader,
                                save_fig=False, set='test')
    bert_probs = np.array(bert_probs) # for easier slicing
    young_probs = bert_probs[:,0]
    old_probs = bert_probs[:, 1]

    # LJ: append young and old probabilities to dataframe
    gen_text_df.insert(len(gen_text_df.columns), 'young_prob', young_probs)
    gen_text_df.insert(len(gen_text_df.columns), 'old_prob', old_probs)

    # LJ: compute and append perplexity column
    gen_text_df['perplexity'] = gen_text_df['text'].apply(perplexity, args=(device,))

    # LJ: compute and append columns for normalized number of distinct n-grams for n = [1,2,3]
    for n in [1, 2, 3]:

        gen_text_df[f'dist_{n}'] = gen_text_df['text'].apply(dist_n_grams, args=(n,))


    # LJ: output csv file name
    attr_model = 'bow' if bag_of_words else 'discrim'
    if bag_of_words:
        wordlist = bag_of_words[20 : len(bag_of_words) - 4]
    else:
        wordlist = 'NA'

    age_group = 'young' if class_label == 0 else 'old'

    # "young_prompt", "neutral_prompt",
    # "old_prompt", "unknown_prompt", "unprompted"

    if prompt_type == "young_prompt":
        prompt_tag = "young"
    elif prompt_type == "neutral_prompt":
        prompt_tag = "neutral"
    elif prompt_type == "old_prompt":
        prompt_tag = "old"
    elif prompt_type == "unknown_prompt":
        prompt_tag = "unk"
    elif prompt_type == "unprompted":
        prompt_tag = "unprompted"

    if pretrained_model.__contains__("/"):
        pretrained_model_no_slash = pretrained_model.replace('/', '-')
        if num_iterations == 0 or stepsize == 0:
            output_path = f'plug_play/output/{prompt_type}/{pretrained_model_no_slash}/ctg_out_am_{attr_model}_pm_{pretrained_model_no_slash}_prompt_{prompt_tag}_wl_{wordlist}_age_NA_WS_baseline.csv'
        else:
            output_path = f'plug_play/output/{prompt_type}/{pretrained_model_no_slash}/ctg_out_am_{attr_model}_pm_{pretrained_model_no_slash}_prompt_{prompt_tag}_wl_{wordlist}_age_{age_group}_WS.csv'
    else:

        if num_iterations == 0 or stepsize == 0:
            output_path = f'plug_play/output/{prompt_type}/{pretrained_model}/ctg_out_am_{attr_model}_pm_{pretrained_model}_prompt_{prompt_tag}_wl_{wordlist}_age_NA_WS_baseline.csv'
        else:
            output_path = f'plug_play/output/{prompt_type}/{pretrained_model}/ctg_out_am_{attr_model}_pm_{pretrained_model}_prompt_{prompt_tag}_wl_{wordlist}_age_{age_group}_WS.csv'


    # create csv file with header if non-existent, append if already exists
    gen_text_df.to_csv(
        output_path,
        index=False,
        mode='a',
        header=not os.path.exists(output_path)
    )

    #######################



    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pretrained_model",
        "-M",
        type=str,
        default="gpt2-medium",
        help="pretrained model name or path to local checkpoint",
    )
    parser.add_argument(
        "--cond_text", type=str, default="The lake",
        help="Prefix texts to condition on"
    )
    parser.add_argument(
        "--uncond", action="store_true",
        help="Generate from end-of-text as prefix"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of samples to generate from the modified latents",
    )
    parser.add_argument(
        "--bag_of_words",
        "-B",
        type=str,
        default=None,
        help="Bags of words used for PPLM-BoW. "
             "Either a BOW id (see list in code) or a filepath. "
             "Multiple BoWs separated by ;",
    )
    parser.add_argument(
        "--discrim",
        "-D",
        type=str,
        default=None,
        choices=("clickbait", "sentiment", "toxicity", "generic"),
        help="Discriminator to use",
    )
    parser.add_argument('--discrim_weights', type=str, default=None,
                        help='Weights for the generic discriminator')
    parser.add_argument('--discrim_meta', type=str, default=None,
                        help='Meta information for the generic discriminator')
    parser.add_argument(
        "--class_label",
        type=int,
        default=-1,
        help="Class label used for the discriminator",
    )
    parser.add_argument("--length", type=int, default=100)
    parser.add_argument("--stepsize", type=float, default=0.02)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument(
        "--sample", action="store_true",
        help="Generate from end-of-text as prefix"
    )
    parser.add_argument("--num_iterations", type=int, default=3)
    parser.add_argument("--grad_length", type=int, default=10000)
    parser.add_argument(
        "--window_length",
        type=int,
        default=0,
        help="Length of past which is being optimized; "
             "0 corresponds to infinite window length",
    )
    parser.add_argument(
        "--horizon_length",
        type=int,
        default=1,
        help="Length of future to optimize over",
    )
    parser.add_argument("--decay", action="store_true",
                        help="whether to decay or not")
    parser.add_argument("--gamma", type=float, default=1.5)
    parser.add_argument("--gm_scale", type=float, default=0.9)
    parser.add_argument("--kl_scale", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no_cuda", action="store_true", help="no cuda")
    parser.add_argument("--colorama", action="store_true",
                        help="colors keywords")
    parser.add_argument("--verbosity", type=str, default="very_verbose",
                        choices=(
                            "quiet", "regular", "verbose", "very_verbose"),
                        help="verbosiry level")
    # Added by LJ
    parser.add_argument("--prompt_type", type=str, default="unknown_prompt",
                        choices=(
                            "young_prompt", "neutral_prompt",
                            "old_prompt", "unknown_prompt", "unprompted"
                        ), help="Type of prompt being used.")

    args = parser.parse_args()
    run_pplm_example(**vars(args))
