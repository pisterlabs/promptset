import os
import logging
import pickle
import hashlib
import time
from pathlib import Path
import sys

import argparse
from langchain.chat_models import ChatOpenAI
from langchain.llms.fake import FakeListLLM
from langchain.prompts import ChatMessagePromptTemplate
from datasets import load_dataset
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).parent.parent))
from eval_llm.ask_llms_examples import ask_positive_and_negative_for_class
from eval_llm.check_by_themselves import check_by_themselves


load_dotenv()

strategy_list = ["normal", "super"]
verification_list = ["dataset", "themselves"]

parser = argparse.ArgumentParser()
parser.add_argument("--group_id", default="default", type=str, help="Group ID of the experiment")
parser.add_argument("--n_trials", default=1, type=int, help="Number of trials")
parser.add_argument("--n_sample_from", default=5, type=int, help="Number of samples from")
parser.add_argument("--n_sample_to", default=5, type=int, help="Number of samples to")
parser.add_argument("--n_sample_step", default=5, type=int, help="Number of samples step")
parser.add_argument("--model", default="gpt-3.5-turbo", type=str, help="Model name", choices=["gpt-3.5-turbo"])
parser.add_argument("--logging", default=True, type=bool, help="Logging to stdout")
parser.add_argument("--max_retry", default=3, type=int, help="Max retry to invoke llms")
parser.add_argument("--test", default=False, type=bool, help="Test mode")
parser.add_argument("--strategy", default="normal", type=str, help="Strategy to ask llms",
                    choices=strategy_list)
parser.add_argument("--verification", default="dataset", type=str, help="Verification method",
                    choices=verification_list)
args = parser.parse_args()

if args.logging:
    print("options", args)

verification = args.verification

is_test = args.test
chat = None
if is_test:
    chat = FakeListLLM(responses=['["yes"]', '["no"]'])
    if args.logging:
        logging.info("Test mode")
else:
    chat = ChatOpenAI(model=args.model, max_retries=args.max_retry)

query_positive = "Please pick up some examples of {label}. You need to pick up {n_examples} examples."
query_negative = "Please pick up some examples which are not {label}. You need to pick up {n_examples} examples."
query_negative_super = "Please pick up some examples which are the superordinate of {label}, but not {label}. You need to pick up {n_examples} examples." # NOQA

if args.strategy == "super":
    pos_q_template = ChatMessagePromptTemplate.from_template(role="user", template=query_positive)
    neg_q_template = ChatMessagePromptTemplate.from_template(role="user", template=query_negative_super)
elif args.strategy == "normal":
    pos_q_template = ChatMessagePromptTemplate.from_template(role="user", template=query_positive)
    neg_q_template = ChatMessagePromptTemplate.from_template(role="user", template=query_negative)
else:
    raise ValueError(f"strategy: {args.strategy} is not supported")

ds_wiki = load_dataset("dbpedia_14", split="train")
n_label = len(ds_wiki.features["label"].names)

n_trials = args.n_trials
n_sample_range = range(args.n_sample_from, args.n_sample_to + 1, args.n_sample_step)

result = []


def get_timestamp():
    return str(time.time()).split('.')[0]


# start experiment
start_time_seconds = get_timestamp()
for n_sample in n_sample_range:
    if args.logging:
        logging.info(f"sample number: {n_sample}")
    for trial_iter in range(1, n_trials+1):
        if args.logging:
            logging.info(f"trial: {trial_iter}/{n_trials}")
        res = []
        if verification == "dataset":
            res = ask_positive_and_negative_for_class(chat, ds_wiki, n_sample, pos_q_template, neg_q_template,
                                                      max_retry=args.max_retry)
        elif verification == "themselves":
            res = check_by_themselves(chat, ds_wiki, n_sample, pos_q_template, neg_q_template,
                                      max_retry=args.max_retry)
        else:
            raise ValueError(f"the way of verification: {verification} is not supported")
        if len(res) != n_label:
            if args.logging:
                logging.info(f"trial: {trial_iter}/{n_trials} partly failed; {len(res)}/{n_label}")
        result.append(res)

# save result into pickle
ex_id = hashlib.md5(str(args).encode()).hexdigest()
datetime = time.strftime("%Y%m%d%H%M%S")
file_name = f"dbpedia_14-{datetime}_{ex_id}.pickle"
file_path = "./results/test/" if is_test else "./results/"
file_path += args.group_id + "/"
os.makedirs(file_path, exist_ok=True)

with open(file_path + file_name, "wb") as f:
    finish_time_seconds = get_timestamp()
    save_data = {
        "start_time": start_time_seconds,
        "finish_time": finish_time_seconds,
        "experiment_id": ex_id,
        "args": args,
        "result": result
    }
    pickle.dump(save_data, f)
    if args.logging:
        logging.info(f"result saved to {file_name}")
