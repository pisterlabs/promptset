# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HugginFace Inc. team.
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
""" OpenAI GPT model fine-tuning script.
    Adapted from https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/train.py
    It self adapted from https://github.com/openai/finetune-transformer-lm/blob/master/train.py

    This script with default values fine-tunes and evaluate a pretrained OpenAI GPT on the RocStories dataset:
        python run_openai_gpt.py \
          --model_name openai-gpt \
          --do_train \
          --do_eval \
          --train_dataset $ROC_STORIES_DIR/cloze_test_val__spring2016\ -\ cloze_test_ALL_val.csv \
          --eval_dataset $ROC_STORIES_DIR/cloze_test_test__spring2016\ -\ cloze_test_ALL_test.csv \
          --output_dir ../log \
          --train_batch_size 16 \
"""
import argparse
import json
import os
import random
import sys
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)

from pytorch_pretrained_bert import OpenAIGPTDoubleHeadsModel, OpenAIGPTTokenizer, OpenAIAdam, cached_path

from SwagExample import read_swag_examples, SwagExample

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

class EncodedSwagExample(object):
    """A single training/test example for the SWAG dataset."""
    def __init__(self,
                 swag_example,
                 tokenizer):
        self.raw_example = swag_example

        self.swag_id = swag_example.swag_id
        self.context_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(swag_example.context_sentence))
        self.start_ending_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(swag_example.start_ending))
        self.endings_tokens = [tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ending)) for ending in swag_example.endings]
        self.label = swag_example.label

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return self.raw_example.__repr__()

def accuracy(out, labels):
    outputs = np.argmax(out, axis=1)
    return np.sum(outputs == labels)

def pre_process_datasets(datasets, input_len, cap_length, start_token, delimiter_token, clf_token):
    """ Pre-process datasets containing lists of SwagExamples

        To Transformer inputs of shape (n_batch, n_endings, length) comprising for each batch, continuation:
        input_ids[batch, ending, :] = [start_token] + context_sent[:cap_length] + [delimiter_token] + start_ending + ending[:cap_length-len(start_ending)] + [clf_token]
    """
    tensor_datasets = []
    n_endings = len(datasets[0][0].raw_example.endings)
    for dataset in datasets:
        n_batch = len(dataset)
        input_ids = np.zeros((n_batch, n_endings, input_len), dtype=np.int64)
        mc_token_ids = np.zeros((n_batch, n_endings), dtype=np.int64)
        lm_labels = np.full((n_batch, n_endings, input_len), fill_value=-1, dtype=np.int64)
        mc_labels = np.zeros((n_batch,), dtype=np.int64)
        for i, swagex, in enumerate(dataset):
            for j, ending_tokens in enumerate(swagex.endings_tokens):
                full_tokens = [start_token] + swagex.context_tokens[:cap_length] + [delimiter_token] + swagex.start_ending_tokens[:cap_length] + ending_tokens[:max(0, cap_length-len(swagex.start_ending_tokens))] + [clf_token]
                input_ids[i, j, :len(full_tokens)] = full_tokens
                mc_token_ids[i, j] = len(full_tokens) - 1
                lm_labels[i, j, :len(full_tokens)-1] = full_tokens[1:]
            mc_labels[i] = swagex.label
        all_inputs = (input_ids, mc_token_ids, lm_labels, mc_labels)
        tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
    return tensor_datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .csv files (or other data files) for the task.")
    parser.add_argument('--model_name', type=str, default='openai-gpt',
                        help='pretrained model name')
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--answer_only", default=False, action='store_true', help="Whether to run with answers only (blank out question).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_from", default=None, type=str,
                        help="The saved model file to load before doing any training or eval (if both --do_train and --do_eval are specified, the saved model will be loaded, then trained, then the trained model will be evaluated).")
    parser.add_argument('--train_filename', type=str, default='train.csv', help="Filename to load train data from (relative to data_dir)")
    parser.add_argument('--eval_filename', type=str, default='val.csv', help="File to load eval data from (relative to data_dir)")
    parser.add_argument('--data_format', type=str, choices=['swag', 'codah'], default='swag', help="Format of the train and eval files (original SWAG CSV format vs our TSV format)")
    parser.add_argument('--model_labels_save_filename', type=str, default='model_labels.json', help="JSON file to save model outputs/labels to (relative to output_dir)")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.002)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lm_coef', type=float, default=0.5)
    parser.add_argument('--n_valid', type=int, default=374)
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=8,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")

    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    args = parser.parse_args()
    print(args)

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    logger.info("device: {}, n_gpu {}".format(device, n_gpu))

    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    if args.do_eval and (not args.do_train) and args.load_model_from is None:
        args.load_model_from = os.path.join(args.output_dir, 'pytorch_model.bin')

    # Load tokenizer and model
    # This loading functions also add new tokens and embeddings called `special tokens`
    # These new embeddings will be fine-tuned on the RocStories dataset
    special_tokens = ['_start_', '_delimiter_', '_classify_']
    tokenizer = OpenAIGPTTokenizer.from_pretrained(args.model_name, special_tokens=special_tokens)
    special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
    model = OpenAIGPTDoubleHeadsModel.from_pretrained(args.model_name, num_special_tokens=len(special_tokens))

    config = model.config
    if args.load_model_from:
        model_state_dict = torch.load(args.load_model_from)
        model = OpenAIGPTDoubleHeadsModel(config)
        model.load_state_dict(model_state_dict)
    model.to(device)

    # Load and encode the datasets
    logger.info("Loading datasets...")
    datasets = []
    dataset_keys = dict()
    if args.do_train:
        train_dataset = read_swag_examples(os.path.join(args.data_dir, args.train_filename), is_training = True, answer_only=args.answer_only, data_format=args.data_format)
        train_dataset = [EncodedSwagExample(ex, tokenizer) for ex in tqdm(train_dataset, desc='Encoding train')]
        dataset_keys['train'] = len(datasets)
        datasets.append(train_dataset)

    if args.do_eval:
        eval_dataset = read_swag_examples(os.path.join(args.data_dir, args.eval_filename), is_training = True, answer_only=args.answer_only, data_format=args.data_format)
        eval_dataset = [EncodedSwagExample(ex, tokenizer) for ex in tqdm(eval_dataset, desc='Encoding eval')]
        dataset_keys['eval'] = len(datasets)
        datasets.append(eval_dataset)

    # Compute the max input length for the Transformer
    max_length = model.config.n_positions // 2 - 2
    input_length = max(len(swagex.context_tokens[:max_length]) + len(swagex.start_ending_tokens[:max_length]) + max(len(ending[:max_length]) for ending in swagex.endings_tokens) + 3  \
                           for dataset in datasets for swagex in dataset)
    input_length = min(input_length, model.config.n_positions)  # Max size of input for the pre-trained model
    print('---')
    print('Input length: {}\n'.format(input_length))
    print('---')

    # Prepare inputs tensors and dataloaders
    tensor_datasets = pre_process_datasets(datasets, input_length, max_length, *special_tokens_ids)
    if args.do_train:
        train_tensor_dataset = tensor_datasets[dataset_keys['train']]
    if args.do_eval:
        eval_tensor_dataset = tensor_datasets[dataset_keys['eval']]

    # Prepare optimizer
    if args.do_train:
        train_data = TensorDataset(*train_tensor_dataset)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        #num_train_optimization_steps = len(train_data) * args.num_train_epochs // args.train_batch_size
        num_train_optimization_steps = int(len(train_data) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        optimizer = OpenAIAdam(optimizer_grouped_parameters,
                               lr=args.learning_rate,
                               warmup=args.warmup_proportion,
                               max_grad_norm=args.max_grad_norm,
                               weight_decay=args.weight_decay,
                               t_total=num_train_optimization_steps)

        nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
        model.train()
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_steps = 0
            tqdm_bar = tqdm(train_dataloader, desc="Training")
            for step, batch in enumerate(tqdm_bar):
                batch = tuple(t.to(device) for t in batch)
                input_ids, mc_token_ids, lm_labels, mc_labels = batch
                losses = model(input_ids, mc_token_ids, lm_labels, mc_labels)
                loss = args.lm_coef * losses[0] + losses[1]

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                nb_tr_steps += 1
                exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
                tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, optimizer.get_lr()[0])

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

    # Save a trained model
    output_model_file = os.path.join(args.output_dir, "pytorch_model.bin")
    if args.do_train:
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        torch.save(model_to_save.state_dict(), output_model_file)

    if args.do_eval:
        eval_data = TensorDataset(*eval_tensor_dataset)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # Load a trained model that you have fine-tuned
        if args.do_train:
            model_state_dict = torch.load(output_model_file)
            model = OpenAIGPTDoubleHeadsModel(config)
            model.load_state_dict(model_state_dict)
            model.to(device)
        model.eval()

        all_model_outputs = []
        data_index = 0

        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, mc_token_ids, lm_labels, mc_labels = batch
            with torch.no_grad():
                _, mc_loss = model(input_ids, mc_token_ids, lm_labels, mc_labels)
                _, mc_logits = model(input_ids, mc_token_ids)

            mc_logits = mc_logits.detach().cpu().numpy()
            mc_labels = mc_labels.to('cpu').numpy()
            tmp_eval_accuracy = accuracy(mc_logits, mc_labels)

            eval_loss += mc_loss.mean().item()
            eval_accuracy += tmp_eval_accuracy

            for i in range(input_ids.size(0)):
                output_obj = dict()
                output_obj['logits'] = [float(x) for x in mc_logits[i]]
                output_obj['true_label'] = int(mc_labels[i])
                output_obj['model_label'] = int(np.argmax(mc_logits[i]))
                output_obj['swag_data'] = datasets[dataset_keys['eval']][data_index].raw_example.to_dict()
                all_model_outputs.append(output_obj)
                data_index += 1

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        eval_loss = eval_loss / nb_eval_steps
        eval_accuracy = eval_accuracy / nb_eval_examples
        train_loss = tr_loss/nb_tr_steps if args.do_train else None
        result = {'eval_loss': eval_loss,
                  'eval_accuracy': eval_accuracy,
                  'train_loss': train_loss}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

        with open(os.path.join(args.output_dir, args.model_labels_save_filename), 'w') as f:
            json.dump(all_model_outputs, f)

if __name__ == '__main__':
    main()
