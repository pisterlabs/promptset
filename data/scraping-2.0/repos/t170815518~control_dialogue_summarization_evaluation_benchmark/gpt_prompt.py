"""
This module is to prompt GPT model with the same prompt as in direct_prompt.py
"""

import sys
import logging
import json
import pickle
import openai
from datasets import load_dataset
import argparse
from datetime import datetime

import tqdm
import wandb
import pandas as pd
import torch
from accelerate import infer_auto_device_map, init_empty_weights
from few_shot_prompt_utility import format_prompt_from_demo_pairs, prompt_llm, evaluate_response_summaries, \
    generate_tf_idf_keywords, generate_control_length, generate_focus_planning

# set up openai api key
openai.api_key = "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"

wandb.login(key='3138e1b24deb278ed045d0dedb39511d3a96245b')

# set up argument parser
parser = argparse.ArgumentParser()
parser.add_argument('-k', type=int, default=1, help='the number of few-shot examples')
parser.add_argument('--demonstration_file', type=str, default=None, help='the pre-generated demonstration file')
parser.add_argument('--dataset', type=str, default='samsum', help='the dataset to evaluate on')
parser.add_argument('--keywords', type=str, default=None, choices=['tfidf'], help='the types of keywords to use')
parser.add_argument('--keyword_num', type=int, default=3, help='the number of keywords to use')
parser.add_argument('--log', type=bool, default=True, help='whether to log the results to wandb')
parser.add_argument('--control', type=str, default=None, choices=['length', 'entity', 'focus'],
                    help='the type of control')
parser.add_argument('--replace_name', type=bool, default=False, help='whether to replace the speaker name with '
                                                                     '#Person1# as DialogSum')
parser.add_argument('--add_instruction', type=bool, default=False)
# parse the arguments
args = parser.parse_args()

# set up log files
logging.basicConfig(
        level=logging.INFO,  # otherwise huggingface has many debug logs
        handlers=[
                logging.FileHandler("{}.log".format(datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))),
                logging.StreamHandler(sys.stdout)
                ]
        )

# start a new wandb run to track this script
if args.log:
    wandb.init(
            project="In-context-learning for Dialogue Summarization",
            # track hyperparameters and run metadata
            config={
                    'model_type': 'GPT3-davinci-003',
                    'k': args.k,
                    'dataset': args.dataset,
                    'keywords': args.keywords,
                    'keyword_num': args.keyword_num,
                    'control': args.control
                    },
            group='performance_in_context_learning',
            job_type='evaluation'
            )

if args.demonstration_file is None and args.k == 0:
    test_dataset = load_dataset(args.dataset, split='test')
    run_id = 0
    results = {0: {}}
    # iterate over test samples to generate k demonstrations
    for test_sample in tqdm.tqdm(test_dataset, total=len(test_dataset)):
        test_id = test_sample['id']
        results[run_id][test_id] = (test_sample, {})
    run_id2demo_pairs = results
else:
    # load the demonstration pickle file
    with open(args.demonstration_file, 'rb') as f:
        run_id2demo_pairs = pickle.load(f)

if args.control == 'entity':
    if args.keywords == 'tfidf':
        run_id2demo_pairs = generate_tf_idf_keywords(run_id2demo_pairs, args.keyword_num)
    else:
        raise NotImplementedError
elif args.control == 'length':
    run_id2demo_pairs = generate_control_length(run_id2demo_pairs)
elif args.control == 'focus':
    assert args.dataset == 'samsum', 'Only samsum dataset has focus control'
    run_id2demo_pairs = generate_focus_planning(run_id2demo_pairs)
else:
    # issue warning
    logging.warning('No control signal is used.')

run_id2prompts, run_id2gold_summaries = format_prompt_from_demo_pairs(run_id2demo_pairs, 'gpt-3', args.replace_name,
                                                                      args.add_instruction,
                                                                      is_focus_planning=args.control == 'focus')

logging.info("Start to prompt the model")
run_id2pred_summaries = {}
run_id2raw_outputs = {}
for run_id, prompts in run_id2prompts.items():
    logging.info("Prompting the model for Run {}".format(run_id))
    run_id2pred_summaries[run_id] = []
    run_id2raw_outputs[run_id] = []
    for prompt in tqdm.tqdm(prompts):
        try:  # prompt the model
            # call GPT model
            if isinstance(prompt, list):
                response = openai.Completion.create(
                        model="text-davinci-003",
                        prompt=prompt[0],
                        max_tokens=100,
                        )
            else:
                response = openai.Completion.create(
                        model="text-davinci-003",
                        prompt=prompt,
                        max_tokens=100,
                        )
            response = response['choices'][0]['text']
        except Exception as e:  # in case any error happens
            logging.info("Exception: {}".format(e))
            logging.info("Prompt: {}".format(prompt))
            response = None
        run_id2pred_summaries[run_id].append(response)
    break  # only run one run for now

logging.info("Start to evaluate the performance")
summary_table, summary_text_table = evaluate_response_summaries(run_id2pred_summaries, run_id2gold_summaries,
                                                                run_id2prompts)

# save the summary table to wandb
summary_table = pd.DataFrame(summary_table)
summary_text_table = pd.DataFrame(summary_text_table)
summary_table = wandb.Table(dataframe=summary_table)
summary_text_table = wandb.Table(dataframe=summary_text_table)
wandb.log({"Evaluation metrics Table": summary_table})
wandb.log({"Summaries Table": summary_text_table})
wandb.finish()
