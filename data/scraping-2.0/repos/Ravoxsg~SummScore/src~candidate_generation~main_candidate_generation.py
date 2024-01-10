# Generate summary candidates with the fine-tuned models.

import os
import time
import argparse
import sys
import torch
import pickle
import datasets
import openai
from tqdm import tqdm
sys.path.append("/data/mathieu/SummScore/src/") # todo: change to your folder path

from common.utils import seed_everything, boolean_string
from common.evaluation import overall_eval 
from model_utils import build_tokenizer, build_model
from dataset import Dataset
from engine import beam_search_step


openai.api_key = "xxx" # todo: fill in your OpenAI key here!!

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--cuda', type = boolean_string, default = True)
parser.add_argument('--debug', type = boolean_string, default = False)
parser.add_argument('--debug_size', type = int, default = 10)
parser.add_argument('--few_shot', type = boolean_string, default = True)

# data
parser.add_argument('--dataset_key', type = str, default = "samsum", choices= ["cnndm", "xsum", "wikihow", "samsum"])

# model
parser.add_argument('--model_type', type = str, default = "pegasus", choices=["pegasus", "bart", "chatgpt"])
parser.add_argument('--model_name', type = str, default = "google/pegasus-large,pegasus_unsupervised",
                    choices = [
                        # Use case #1: Unsupervised abstractive summarization
                        "google/pegasus-large,pegasus_unsupervised", "gpt-3.5-turbo,chatgpt",

                        # Use case #2: Zero-shot transfer
                        # from CNN/DM
                        "google/pegasus-cnn_dailymail,pegasus_cnndm", "facebook/bart-large-cnn,bart_cnndm", "Yale-LILY/brio-cnndm-cased,brio_cnndm",
                        # from XSum
                        "google/pegasus-xsum,pegasus_xsum", "facebook/bart-large-xsum,bart_xsum", "Yale-LILY/brio-xsum-cased,brio_xsum",
                        # from WikiHow
                        "google/pegasus-wikihow,pegasus_wikihow", "our_bart_wikihow,bart_wikihow",
                        # from SAMSum
                        "our_pegasus_samsum,pegasus_samsum", "our_bart_samsum,bart_samsum"
                    ])
parser.add_argument('--hidden_size', type = int, default = 768) # 768
parser.add_argument('--cache_dir', type = str, default = "../../../hf_models/pegasus-large/")
parser.add_argument('--load_model_path', type = str, default = "finetuned_checkpoints/our_pegasus_samsum.pt")

# summary generation
parser.add_argument('--val_dataset', type = str, default = "val", choices = ["val", "test"])
parser.add_argument('--max_val_size', type = int, default = 1000)
parser.add_argument('--inference_bs', type = int, default = 6)
parser.add_argument('--save_summaries', type = boolean_string, default = True)
parser.add_argument('--generation_method', type = str, default = "beam_search",
                    choices = ["beam_search", "diverse_beam_search", "top_p_sampling", "top_k_sampling"])
parser.add_argument('--num_return_sequences', type = int, default = 20) # default: 15
parser.add_argument('--num_beams', type = int, default = 20) # for beam search
parser.add_argument('--num_beam_groups', type = int, default = 20) # for diverse beam search
parser.add_argument('--diversity_penalty', type = float, default = 1.0) # for diverse beam search
parser.add_argument('--top_p', type = float, default = 0.95) # for top-p sampling
parser.add_argument('--top_k', type = int, default = 50) # for top-k sampling
parser.add_argument('--repetition_penalty', type = float, default = 1.0) # for diverse beam search
parser.add_argument('--stemmer', type = boolean_string, default = True)

# metrics
parser.add_argument('--eval_rouge', type = boolean_string, default = True)
parser.add_argument('--eval_bertscore', type = boolean_string, default = False)
parser.add_argument('--eval_new_ngram', type = boolean_string, default = True)
parser.add_argument('--eval_rouge_text', type = boolean_string, default = False)
parser.add_argument('--n_show_summaries', type = int, default=0)

args = parser.parse_args()


dataset_keys = ["cnndm", "xsum", "wikihow", "samsum"]
dataset_names = ["ccdv/cnn_dailymail", "xsum", "wikihow", "samsum"]
dataset_versions = ["3.0.0", "default", "all", "samsum"]
text_keys = ["article", "document", "text", "dialogue"]
summary_keys = ["highlights", "summary", "headline", "summary"]
max_lengths = [1024, 512, 512, 512]
max_summary_lengths = [128, 64, 128, 64]
length_penalties_pegasus = [0.8, 0.8, 0.6, 0.8]
length_penalties_bart = [1.0, 1.0, 1.0, 1.0]
no_repeat_ngram_sizes_pegasus = [0, 3, 0, 0]
no_repeat_ngram_sizes_bart = [3, 3, 3, 3]
ns = [3, 1, 3, 2]

idx = dataset_keys.index(args.dataset_key)

args.dataset_name = dataset_names[idx]
args.dataset_version = dataset_versions[idx]
args.text_key = text_keys[idx]
args.summary_key = summary_keys[idx]
args.max_length = max_lengths[idx]
args.max_summary_length = max_summary_lengths[idx]
if args.model_type == "pegasus":
    args.length_penalty = length_penalties_pegasus[idx]
    args.no_repeat_ngram_size = no_repeat_ngram_sizes_pegasus[idx]
    if "unsupervised" in args.model_name:
        args.no_repeat_ngram_size = 3
elif args.model_type == "bart":
    args.length_penalty = length_penalties_bart[idx]
    args.no_repeat_ngram_size = no_repeat_ngram_sizes_bart[idx]
args.n = ns[idx]

model_name = args.model_name.split(",")
args.model_name = model_name[0]
args.clean_model_name = model_name[1]

print("*"*50)
print(args)


def main(args):
    # seed
    seed_everything(args.seed)

    os.makedirs(f"../../summaries/{args.dataset_key}/{args.val_dataset}/{args.generation_method}/", exist_ok=True)

    # device
    device = torch.device("cpu")
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    args.device = device
    print("\nUsing device {}".format(device))

    # data
    dataset_args = [args.dataset_name, args.dataset_version]
    data = datasets.load_dataset(*dataset_args)
    val_data = data["validation"]
    if args.val_dataset == "test":
        val_data = data["test"]
    texts = [val_data[i][args.text_key] for i in range(len(val_data))]
    labels = [val_data[i][args.summary_key] for i in range(len(val_data))]

    # permute
    p = pickle.load(open(f"../{args.val_dataset}_permutations/{args.dataset_name}_{args.val_dataset}_permutation.pkl", "rb"))
    print(p[:10])
    texts = [texts[x] for x in p]
    labels = [labels[x] for x in p]

    # sample
    if args.val_dataset != "test" or args.model_type == "chatgpt":
        texts = texts[:args.max_val_size]
        labels = labels[:args.max_val_size]
    if args.debug:
        texts = texts[:args.debug_size]
        labels = labels[:args.debug_size]

    # run the inference
    print(f"Running inference on {len(texts)} data points...")
    summaries = []
    # you might have to re-run the script 2-3 times to be able to generate summaries on all datapoints
    if args.model_type == "chatgpt":
        failed_index = []
        for i in tqdm(range(len(texts))):
            text = texts[i]
            prompt = f"Text: {text}.\nSummarize the above text in {args.n} sentence."
            try:
                responsefromgpt = openai.ChatCompletion.create(
                    model=f"{args.gpt_model}",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=args.max_summary_length,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    n=args.num_beams
                )
                summaries_i = [responsefromgpt['choices'][j]['message']['content'] for j in range(args.num_beams)]
            except Exception as exc:
                failed_index.append(i)
                summaries_i = []
            summaries.append(summaries_i)
    else:
        # tokenizer
        tokenizer = build_tokenizer(args)
        # datasets
        dataset = Dataset(tokenizer, texts, labels, args)
        print("Total size of dataset: {}".format(len(texts)))
        # data loader
        loader = torch.utils.data.DataLoader(dataset, batch_size = args.inference_bs, shuffle = False)
        # model
        model = build_model(args)
        model = model.to(args.device)
        # loop
        for idx, batch in tqdm(enumerate(loader)):
            batch["text_inputs"]["input_ids"] = batch["text_inputs"]["input_ids"].squeeze(1).to(args.device)
            batch["text_inputs"]["attention_mask"] = batch["text_inputs"]["attention_mask"].squeeze(1).to(args.device)
            summaries_i = beam_search_step(batch, tokenizer, model, args)
            summaries += summaries_i
            if idx == 0:
                print("*"*50)
                print(batch["text"][0])
                print("*"*20)
                print(summaries_i[0])

    # evaluation
    base_results = [summaries[i][0] for i in range(len(summaries))]
    print("*"*100)
    print("\nTop beam:")
    overall_eval(texts, base_results, labels, args)

    # export
    num_candidates = len(summaries[0])
    if args.save_summaries:
        path = f"../../summaries/{args.dataset_key}/{args.val_dataset}/{args.generation_method}/"
        with open(path + f"{args.val_dataset}_texts_{len(texts)}_beams_{num_candidates}.pkl", "wb") as f:
            pickle.dump(texts, f)
        with open(path + f"{args.val_dataset}_summaries_{args.clean_model_name}_{len(texts)}_beams_{num_candidates}.pkl", "wb") as f:
            pickle.dump(summaries, f)
        with open(path + f"{args.val_dataset}_labels_{len(texts)}_beams_{num_candidates}.pkl", "wb") as f:
            pickle.dump(labels, f)
        print("saved generated summaries!", path)


if __name__ == '__main__':

    main(args)
