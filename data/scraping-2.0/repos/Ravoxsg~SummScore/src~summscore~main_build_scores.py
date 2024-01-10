# Generate summary candidates with the fine-tuned models.

import os
import numpy as np
import argparse
import sys
import torch
import pickle
import datasets
import openai
from tqdm import tqdm
sys.path.append("/data/mathieu/SummScore/src/") # todo: change to your folder path

from common.utils import seed_everything, boolean_string
from engine import build_scores


openai.api_key = "xxx" # todo: fill in your OpenAI key here!!

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--cuda', type = boolean_string, default = True)
parser.add_argument('--debug', type = boolean_string, default = False)
parser.add_argument('--debug_size', type = int, default = 10)
parser.add_argument('--few_shot', type = boolean_string, default = True)

# data
parser.add_argument('--dataset_key', type = str, default = "samsum", choices= ["cnndm", "xsum", "wikihow", "samsum"])
parser.add_argument('--generation_methods', type = list, default = [
    "beam_search",
    #"diverse_beam_search",
    #"top_p_sampling",
    #"top_k_sampling",
])

# model
parser.add_argument('--model_type', type = str, default="pegasus", choices=["pegasus","bart"])
parser.add_argument('--clean_model_name', type = str, default = "pegasus_unsupervised",
                    choices = [
                        # Use case #1: Unsupervised abstractive summarization
                        "pegasus_unsupervised", "chatgpt",

                        # Use case #2: Zero-shot transfer
                        # from CNN/DM
                        "pegasus_cnndm", "bart_cnndm", "brio_cnndm",
                        # from XSum
                        "pegasus_xsum", "bart_xsum", "brio_xsum",
                        # from WikiHow
                        "pegasus_wikihow", "bart_wikihow",
                        # from SAMSum
                        "pegasus_samsum", "bart_samsum"
                    ])

# summary generation
parser.add_argument('--val_dataset', type = str, default = "val", choices = ["val", "test"])
parser.add_argument('--max_val_size', type = int, default = 1000)
parser.add_argument('--num_beams', type = int, default = 20) # for beam search

# features for SummScore
parser.add_argument('--metrics_to_use', type = dict, default = {
    # n-gram overlap with the source
    "rouge_1": 1.0,
    "rouge_2": 1.0,
    "bleu": 1.0,
    # semantic similarity with the source
    "bert_score": 1.0,
    "bart_score": 1.0,
    "bleurt": 1.0,
    # intrinsic summary quality
    "diversity": 1.0,
    "length": 1.0,
})
parser.add_argument('--compute_rouge', type = boolean_string, default = True)
parser.add_argument('--compute_bleu', type = boolean_string, default = True)
parser.add_argument('--compute_bertscore', type = boolean_string, default = True)
parser.add_argument('--efficient_bertscore', type = boolean_string, default = False)
parser.add_argument('--n_efficient', type = int, default = 10)
parser.add_argument('--compute_bartscore', type = boolean_string, default = True)
parser.add_argument('--compute_bleurt', type = boolean_string, default = True)
parser.add_argument('--compute_diversity', type = boolean_string, default = True)
parser.add_argument('--compute_length', type = boolean_string, default = True)
parser.add_argument('--stemmer', type = boolean_string, default = True)

args = parser.parse_args()

dataset_keys = ["cnndm", "xsum", "wikihow", "samsum"]
val_sizes = [13368, 11332, 5600, 818]
test_sizes = [11490, 11334, 5580, 819]
ratios =  [60.8, 23.21, 23.28, 62.08, 23.42]

idx = dataset_keys.index(args.dataset_key)

if args.val_dataset == "val":
    args.val_size = val_sizes[idx]
elif args.val_dataset == "test":
    args.val_size = test_sizes[idx]
args.ratio = ratios[idx]

print("*"*50)
print(args)


def main(args):
    # seed
    seed_everything(args.seed)

    # device
    device = torch.device("cpu")
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    args.device = device
    print(f"Using device: {device}")

    # load data
    size = min(args.val_size, args.max_val_size)
    path = f"../../summaries/{args.dataset_key}/{args.val_dataset}/{args.generation_methods[0]}/"
    texts_path = path + f"{args.val_dataset}_texts_{size}_beams_{args.num_beams}.pkl"
    texts = pickle.load(open(texts_path, "rb"))
    summaries_path = path + f"{args.val_dataset}_summaries_{args.clean_model_name}_{size}_beams_{args.num_beams}.pkl"
    summaries = pickle.load(open(summaries_path, "rb"))

    # build the scores for each summary candidate
    all_scores = build_scores(texts, summaries, args)


if __name__ == '__main__':
    main(args)
