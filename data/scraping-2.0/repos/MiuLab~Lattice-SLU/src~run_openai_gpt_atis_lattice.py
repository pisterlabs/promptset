# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
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

import argparse
import os
import csv
import random
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from sklearn.metrics import f1_score
from pytorch_pretrained_bert import OpenAIAdam, cached_path, WEIGHTS_NAME, CONFIG_NAME
from modeling_openai import OpenAIGPTDoubleHeadsClsModel
from tokenization_openai import OpenAIGPTTokenizer
from lattice_utils import LatticeReader, LatticeNode, Lattice

ROCSTORIES_URL = "https://s3.amazonaws.com/datasets.huggingface.co/ROCStories.tar.gz"
LABEL_FILES = {
    "intent": "atis_intent_labels.txt",
    "slot": "atis_slot_labels.txt"
}

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)


def accuracy(out, labels):
    outputs = (out >= 0.5).astype(np.int32)
    acc = 0
    for o, l in zip(outputs, labels.astype(np.int32)):
        if np.all(o == l):
            acc += 1
    return acc

def f1(out, labels):
    outputs = (torch.tensor(out).sigmoid().numpy() >= 0.5).astype(np.int32)
    labels = labels.astype(np.int32)
    return f1_score(labels, outputs, average='micro')

def load_atis_dataset(dataset_path, label_list, tokenizer, probabilistic_masks=True, linearize=False, plot=False):
    """ Output a list of tuples(story, 1st continuation, 2nd continuation, label) """
    with open(dataset_path, encoding='utf_8') as csvfile:
        reader = csv.DictReader(csvfile)
        output = []
        latt_reader = LatticeReader()
        for i, row in enumerate(tqdm(reader)):
            labels = row["labels"].strip().split('|')
            labels = [label_list.index(l) for l in labels]
            # print(row["text"])
            lattice = latt_reader.read_sent(row["text"], i)
            if plot:
                lattice.plot(f"lattices/{i}", show_log_probs=["fwd_log_prob"])
            utt = " ".join(lattice.str_tokens())
            tokens, mapping = tokenizer.tokenize_with_map(utt)
            nodes = []
            node_map = {}
            for j, (node_idx, n) in enumerate(mapping):
                if node_idx not in node_map: node_map[node_idx] = []
                node_map[node_idx].append(j)
            for token, (node_idx, n) in zip(tokens, mapping):
                orig_node = lattice.nodes[node_idx]
                if n != 0:
                    nodes_prev = [len(nodes)-1]
                else:
                    nodes_prev = [node_map[node][-1] for node in orig_node.nodes_prev]
                if n != len(node_map[node_idx])-1:
                    nodes_next = [len(nodes)+1]
                else:
                    nodes_next = [node_map[node][0] for node in orig_node.nodes_next]
                node = LatticeNode(nodes_prev=nodes_prev, nodes_next=nodes_next, value=token,
                                   fwd_log_prob=orig_node.fwd_log_prob if n == 0 else 0,
                                   marginal_log_prob=orig_node.marginal_log_prob,
                                   bwd_log_prob=orig_node.bwd_log_prob if n == len(node_map[node_idx])-1 else 0)
                nodes.append(node)
            lattice_cut = Lattice(idx=i, nodes=nodes)
            if linearize:
                pos = [i for i in range(len(lattice_cut.nodes))]
            else:
                pos = lattice_cut.longest_distances()
                pos = [p+1 for p in pos]
            log_conditional = lattice_cut.compute_pairwise_log_conditionals("bwd", probabilistic_masks)[0]
            marginals = np.array([node.marginal_log_prob for node in lattice_cut.nodes])
            output.append((utt, pos, log_conditional, marginals, labels))
    return output

def pre_process_datasets(encoded_datasets, input_len, cap_length, start_token, delimiter_token, clf_token, num_labels):
    """ Pre-process datasets containing lists of tuples(story, 1st continuation, 2nd continuation, label)

        To Transformer inputs of shape (n_batch, n_alternative, length) comprising for each batch, continuation:
        input_ids[batch, alternative, :] = [start_token] + story[:cap_length] + [delimiter_token] + cont1[:cap_length] + [clf_token]
    """
    tensor_datasets = []
    for dataset in encoded_datasets:
        n_batch = len(dataset)
        input_ids = np.zeros((n_batch, input_len), dtype=np.int64)
        token_ids = np.zeros((n_batch,), dtype=np.int64)
        pos_ids = np.zeros((n_batch, input_len), dtype=np.int64)
        log_conditionals = np.zeros((n_batch, input_len, input_len), dtype=np.float32)
        lm_labels = np.full((n_batch, input_len), fill_value=-1, dtype=np.int64)
        labels = np.zeros((n_batch, num_labels), dtype=np.float32)
        for i, (utt, pos, log_conditional, marginals, label), in enumerate(dataset):
            sent = [start_token] + utt[:cap_length] + [clf_token]
            input_ids[i, :len(sent)] = sent
            positions = [0] + pos[:cap_length] + [max(pos[:cap_length])+1]
            pos_ids[i, :len(positions)] = positions
            log_conditionals[i, 1:(log_conditional.shape[0]+1), 1:(log_conditional.shape[1]+1)] = log_conditional
            token_ids[i] = len(sent) - 1
            lm_labels[i, :len(sent)] = sent
            for l in label:
                labels[i, l] = 1.0
        all_inputs = (input_ids, token_ids, pos_ids, log_conditionals, lm_labels, labels)
        tensor_datasets.append(tuple(torch.tensor(t) for t in all_inputs))
    return tensor_datasets

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='openai-gpt',
                        help='pretrained model name')
    parser.add_argument('--task', type=str, default='intent',
                        choices=['intent', 'slot'], help="Intent or slot prediction")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument('--train_dataset', type=str, default='')
    parser.add_argument('--eval_dataset', type=str, default='')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_train_epochs', type=int, default=3)
    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--eval_batch_size', type=int, default=16)
    parser.add_argument('--max_grad_norm', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=6.25e-5)
    parser.add_argument('--warmup_proportion', type=float, default=0.002)
    parser.add_argument('--lr_schedule', type=str, default='warmup_linear')
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--lm_coef', type=float, default=0.0)
    parser.add_argument('--probabilistic_masks', action='store_true')
    parser.add_argument('--attn_bias', action='store_true')
    parser.add_argument('--linearize', action='store_true')
    args = parser.parse_args()
    print(args)

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

    label_list = list()
    for line in open(LABEL_FILES[args.task]):
        label_list.append(line.strip())

    # Load tokenizer and model
    # This loading functions also add new tokens and embeddings called `special tokens`
    # These new embeddings will be fine-tuned on the RocStories dataset
    special_tokens = ['_start_', '_delimiter_', '_classify_']
    tokenizer = OpenAIGPTTokenizer.from_pretrained(args.model_name, special_tokens=special_tokens)
    special_tokens_ids = list(tokenizer.convert_tokens_to_ids(token) for token in special_tokens)
    model = OpenAIGPTDoubleHeadsClsModel.from_pretrained(args.model_name, num_labels=len(label_list), num_special_tokens=len(special_tokens))
    model.to(device)

    # Load and encode the datasets
    def tokenize_and_encode(obj):
        """ Tokenize and encode a nested object """
        if isinstance(obj, str):
            return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
        elif isinstance(obj, int):
            return obj
        elif isinstance(obj, np.ndarray):
            return obj
        return list(tokenize_and_encode(o) for o in obj)

    logger.info("Encoding dataset...")
    train_dataset = load_atis_dataset(args.train_dataset, label_list, tokenizer, args.probabilistic_masks, args.linearize)
    eval_dataset = load_atis_dataset(args.eval_dataset, label_list, tokenizer, args.probabilistic_masks, args.linearize, plot=False)
    datasets = (train_dataset, eval_dataset)
    encoded_datasets = tokenize_and_encode(datasets)

    # Compute the max input length for the Transformer
    max_length = model.config.n_positions - 2
    input_length = max(len(utt[:max_length]) + 2  \
                           for dataset in encoded_datasets for utt, _, _, _, _ in dataset)
    input_length = min(input_length, model.config.n_positions)  # Max size of input for the pre-trained model

    # Prepare inputs tensors and dataloaders
    tensor_datasets = pre_process_datasets(encoded_datasets, input_length, max_length, *special_tokens_ids, len(label_list))
    train_tensor_dataset, eval_tensor_dataset = tensor_datasets[0], tensor_datasets[1]

    train_data = TensorDataset(*train_tensor_dataset)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_data = TensorDataset(*eval_tensor_dataset)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Prepare optimizer
    if args.do_train:
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        num_train_optimization_steps = len(train_dataloader) * args.num_train_epochs
        optimizer = OpenAIAdam(optimizer_grouped_parameters,
                               lr=args.learning_rate,
                               warmup=args.warmup_proportion,
                               max_grad_norm=args.max_grad_norm,
                               weight_decay=args.weight_decay,
                               t_total=num_train_optimization_steps)

    if args.do_train:
        nb_tr_steps, tr_loss, exp_average_loss = 0, 0, None
        model.train()
        results = []
        for _ in trange(int(args.num_train_epochs), desc="Epoch"):
            tr_loss = 0
            nb_tr_steps = 0
            tqdm_bar = tqdm(train_dataloader, desc="Training")
            for step, batch in enumerate(tqdm_bar):
                batch = tuple(t.to(device) for t in batch)
                input_ids, mc_token_ids, pos_ids, attn_bias, lm_labels, mc_labels = batch
                losses = model(input_ids, mc_token_ids, lm_labels, mc_labels, position_ids=pos_ids, attn_bias=attn_bias if args.attn_bias else None)
                # loss = args.lm_coef * losses[0] + losses[1]
                loss = losses[1]
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                tr_loss += loss.item()
                exp_average_loss = loss.item() if exp_average_loss is None else 0.7*exp_average_loss+0.3*loss.item()
                nb_tr_steps += 1
                tqdm_bar.desc = "Training loss: {:.2e} lr: {:.2e}".format(exp_average_loss, optimizer.get_lr()[0])

            model.eval()
            eval_loss = 0
            nb_eval_steps, nb_eval_examples = 0, 0
            all_logits, all_labels = [], []
            for batch in tqdm(eval_dataloader, desc="Evaluating"):
                batch = tuple(t.to(device) for t in batch)
                input_ids, mc_token_ids, pos_ids, attn_bias, lm_labels, mc_labels = batch
                with torch.no_grad():
                    _, mc_loss = model(input_ids, mc_token_ids, lm_labels, mc_labels, position_ids=pos_ids, attn_bias=attn_bias if args.attn_bias else None)
                    _, mc_logits = model(input_ids, mc_token_ids, position_ids=pos_ids, attn_bias=attn_bias if args.attn_bias else None)

                mc_logits = mc_logits.detach().cpu().numpy()
                mc_labels = mc_labels.to('cpu').numpy()

                eval_loss += mc_loss.mean().item()
                all_logits.append(mc_logits)
                all_labels.append(mc_labels)

                nb_eval_examples += input_ids.size(0)
                nb_eval_steps += 1

            eval_loss = eval_loss / nb_eval_steps
            all_logits = np.concatenate(all_logits, axis=0)
            all_labels = np.concatenate(all_labels, axis=0)
            eval_f1 = f1(all_logits, all_labels)
            eval_acc = accuracy(all_logits, all_labels) / nb_eval_examples
            train_loss = tr_loss/nb_tr_steps if args.do_train else None
            result = {'eval_loss': eval_loss,
                      'eval_f1': eval_f1,
                      'eval_accuracy': eval_acc,
                      'train_loss': train_loss}
            print(result)
            results.append(result)

        with open(os.path.join(args.output_dir, "log.csv"), "w") as csvfile:
            writer = csv.DictWriter(
                csvfile,
                ["train_loss", "eval_loss", "eval_accuracy", "eval_f1"]
            )
            writer.writeheader()
            writer.writerows(results)

    # Save a trained model
    if args.do_train:
        # Save a trained model, configuration and tokenizer
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        # Load a trained model and vocabulary that you have fine-tuned
        model = OpenAIGPTDoubleHeadsClsModel.from_pretrained(
            args.output_dir,num_labels=len(label_list), num_special_tokens=len(special_tokens))
        tokenizer = OpenAIGPTTokenizer.from_pretrained(args.output_dir)
        model.to(device)

    if args.do_eval:
        model.eval()
        eval_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        all_logits, all_labels = [], []
        fw = open("prediction.txt", "w")
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(device) for t in batch)
            input_ids, mc_token_ids, pos_ids, attn_bias, lm_labels, mc_labels = batch
            with torch.no_grad():
                _, mc_loss = model(input_ids, mc_token_ids, lm_labels, mc_labels, position_ids=pos_ids, attn_bias=attn_bias if args.attn_bias else None)
                _, mc_logits = model(input_ids, mc_token_ids, position_ids=pos_ids, attn_bias=attn_bias if args.attn_bias else None)

            mc_logits = mc_logits.detach().cpu().numpy()
            mc_labels = mc_labels.to('cpu').numpy()

            for i, (o, l) in enumerate(zip((mc_logits>=0.5).astype(np.int32), mc_labels.astype(np.int32))):
                # if np.any(o != l):
                # pred = [label_list[idx] for idx, val in enumerate(o) if val == 1]
                # true = [label_list[idx] for idx, val in enumerate(l) if val == 1]
                pred = o
                true = l
                fw.write(f"{eval_dataset[nb_eval_examples+i][0]}\n{pred}\n{true}\n\n")

            eval_loss += mc_loss.mean().item()
            all_logits.append(mc_logits)
            all_labels.append(mc_labels)

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        fw.close()
        eval_loss = eval_loss / nb_eval_steps
        all_logits = np.concatenate(all_logits, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        eval_f1 = f1(all_logits, all_labels)
        eval_acc = accuracy(all_logits, all_labels) / nb_eval_examples
        train_loss = tr_loss/nb_tr_steps if args.do_train else None
        result = {'eval_loss': eval_loss,
                  'eval_f1': eval_f1,
                  'eval_accuracy': eval_acc,
                  'train_loss': train_loss}

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

if __name__ == '__main__':
    main()
