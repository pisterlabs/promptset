
import os
import openai

from argparse import ArgumentParser
from tqdm import tqdm
import csv
import re
import random

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import json
import pickle

openai.api_key = os.getenv("OPENAI_API_KEY")

def query_gpt_completion(prompt_ls, queries, uuids, ground_truth_answers, pred_dir, pred_file_name, max_prompt_size=300, temperature=0, max_tokens=100):
    pred_response_path = os.path.join(pred_dir, f"{pred_file_name}_response.pkl")
    pred_result_path =  os.path.join(pred_dir, f"{pred_file_name}_result.json")
    pred_response_string_path = os.path.join(pred_dir, f"{pred_file_name}_response_strings.json")

    if os.path.exists(pred_result_path):
      y_preds = json.load(open(pred_result_path))
      y_preds = [pred for _, pred in y_preds]
      return y_preds

    # Make sure the prompt is not too large
    prompt_size = max([len(prompt.split()) for prompt in prompt_ls])
    assert prompt_size < max_prompt_size

    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt_ls,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )

    y_pred_strings = [response["choices"][i]["text"] for i in range(len(prompt_ls))]

    response_pairs = [t for t in zip(uuids, response)]
    response_string_tpls = [t for t in zip(uuids, y_pred_strings, queries, ground_truth_answers)]
    result_pairs = [t for t in zip(uuids, y_pred_strings)]

    pickle.dump(response_pairs, open(pred_response_path, 'wb'))
    json.dump(response_string_tpls, open(pred_response_string_path, 'w'))
    json.dump(result_pairs, open(pred_result_path, 'w'))
    return y_pred_strings

class CLUTRRDataset:
  def __init__(self, root, dataset, split, k):
    self.dataset_dir = os.path.join(root, f"CLUTRR/{dataset}/")
    self.file_names = [os.path.join(self.dataset_dir, d) for d in os.listdir(self.dataset_dir) if f"_{split}.csv" in d]
    if k is not None:
      self.file_names = [x for x in self.file_names if f"1.{k}" in x]
    self.data = [row for f in self.file_names for row in list(csv.reader(open(f)))[1:]]

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    datapoint = self.data[i]
    uuid = datapoint[1]
    # Query is of type (sub, obj)
    query_sub_obj = eval(datapoint[3])

    # Context
    context = datapoint[2].replace("[", "").replace("]", "")
    prompt = context + f" Who is {query_sub_obj[1]} to {query_sub_obj[0]}? Let's think step by step."
    task = datapoint[10]

    # Answer is one of 20 classes such as daughter, mother, ...
    answer = datapoint[5]
    return ((task, uuid), prompt, query_sub_obj, answer)

  @staticmethod
  def collate_fn(batch):
    # Note prompt engineering!
    uuids = [uuid for (uuid, _, _, _) in batch]
    prompts = [prompt for (_, prompt, _, _) in batch]
    queries = [query for (_, _, query, _) in batch]
    answers = [answer for (_, _, _, answer) in batch]
    return (uuids, prompts, queries, answers)

def clutrr_loader(root, dataset, batch_size, k):
  test_dataset = CLUTRRDataset(root, dataset, "test", k)
  test_loader = DataLoader(test_dataset, batch_size, collate_fn=CLUTRRDataset.collate_fn, shuffle=True)
  return test_loader

class Trainer:
  def __init__(self, train_loader, test_loader, device, pred_dir, pred_file_name, learning_rate, **args):
    self.device = device
    self.pred_dir = pred_dir
    self.pred_file_name = pred_file_name
    self.train_loader = train_loader
    self.test_loader = test_loader
    self.min_test_loss = 10000000000.0
    self.max_accu = 0

  def loss(self, y_pred, y):
    (_, dim) = y_pred.shape
    gt = torch.stack([torch.tensor([1.0 if i == t else 0.0 for i in range(dim)]) for t in y])
    return nn.functional.binary_cross_entropy(y_pred, gt)

  def accuracy(self, y_pred, y):
    batch_size = len(y)
    num_correct = len([() for i, j in zip(y_pred, y) if i == j])
    return (num_correct, batch_size)

  def train(self, num_epochs):
    for i in range(1, num_epochs + 1):
      self.train_epoch(i)
      self.test_epoch(i)

  def train_epoch(self, epoch):
    raise NotImplementedError("We don't finetune GPT-3 here.")

  def test_epoch(self, epoch):
    with torch.no_grad():
      iterator = tqdm(self.test_loader)
      for (i, (uuids, xs, queries, ys)) in enumerate(iterator):
        query_gpt_completion(xs, queries, uuids, ys, self.pred_dir,  f"{self.pred_file_name}_{i}")

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("--dataset", type=str, default="data_089907f8")
  parser.add_argument("--pred_file_name", type=str, default="pred_batch")
  parser.add_argument("--pred_dir", type=str, default="gpt_preds_zero_cot_pred_all_1")
  parser.add_argument("--load_model", type=bool, default=False)
  parser.add_argument("--n-epochs", type=int, default=100)
  parser.add_argument("--batch-size", type=int, default=16)
  parser.add_argument("--seed", type=int, default=1234)
  parser.add_argument("--learning-rate", type=float, default=0.00001)
  parser.add_argument("--num-mlp-layers", type=int, default=2)
  parser.add_argument("--no-fine-tune-roberta", type=bool, default=False)
  parser.add_argument("--scallop-softmax", type=bool, default=False)
  parser.add_argument("--debug", action="store_true")
  parser.add_argument("--use-last-hidden-state", action="store_true")
  parser.add_argument("-k", type=int)
  parser.add_argument("--cuda", type=bool, default=True)
  parser.add_argument("--gpu", type=int, default=0)
  args = parser.parse_args()
  print(args)

  # Parameters
  torch.manual_seed(args.seed)
  random.seed(args.seed)
  if args.cuda:
    if torch.cuda.is_available(): device = torch.device(f"cuda:{args.gpu}")
    else: raise Exception("No cuda available")
  else: device = torch.device("cpu")

  # Setting up data and model directories
  data_root = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../data"))
  model_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../../model/clutrr"))
  if not os.path.exists(model_dir): os.makedirs(model_dir)

  args.pred_dir = os.path.join(model_dir, args.pred_dir)
  if not os.path.exists(args.pred_dir): os.makedirs(args.pred_dir)

  # Load the dataset
  test_loader = clutrr_loader(data_root, args.dataset, args.batch_size, args.k)

  # Train
  trainer = Trainer(None, test_loader, device, args.pred_dir, args.pred_file_name, args.learning_rate, num_mlp_layers=args.num_mlp_layers, debug=args.debug, no_fine_tune_roberta=args.no_fine_tune_roberta)
  trainer.test_epoch(0)
  print()
