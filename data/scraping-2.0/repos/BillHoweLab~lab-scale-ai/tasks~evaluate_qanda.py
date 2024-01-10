import evaluate
import numpy as np
import json
import argparse
import torch
import wandb

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from peft import PeftModel
from typing import Iterable
from tqdm import tqdm
from os import path, makedirs, getenv

from openai_chat_api import DialogueBot
from generate_from_hf_model import generate_from_prompt

from collections import Counter
import re
import string
from typing import Optional, List

#####
# TODO: Below is partially adapted better answer parsing from  
# https://github.com/vlievin/medical-reasoning/blob/master/medical_reasoning/models/functional/infer_answer.py
# to be completed, and new metric added.

def parse_options_from_input(input_question: str) -> dict:
    # extract the options part from the input question
    options_str = re.search(r"\{(.+?)\}$", input_question)
    if options_str:
        options_str = options_str.group(1)
        options = dict(item.split(': ') for item in options_str.split(', '))
        return options
    else:
        return {}

def get_start_indices(target: str, pattern: str) -> list[int]:
    try:
        matches = re.finditer(pattern, target)
        return [m.start() for m in matches]
    except Exception as exc:
        return []

def get_first_match(query, choices, keys, op=min):
    assert len(choices) == len(keys)
    indices = [(key, get_start_indices(query, o)) for key, o in zip(keys, choices)]
    indices = list(filter(lambda x: len(x[1]), indices))
    if len(indices):
        return op(indices, key=lambda x: x[1])[0]
    else:
        return None

def infer_answer_from_input(input_question: str, target_answer: str) -> Optional[str]:
    options = parse_options_from_input(input_question)
    if not options:
        return None

    # check if the target answer is directly one of the option keys
    if target_answer.strip() in options:
        return target_answer.strip()

    # direct match with the provided options' values
    for key, value in options.items():
        if value.strip() == target_answer.strip():
            return key

    # use regex patterns to match the answer
    option_symbols = list(options.keys())
    option_values = list(options.values())
    option_symbols_re = [rf"{re.escape(o)}(\)|:|\.|,| )" for o in option_symbols]

    # try to match using option symbols
    match = get_first_match(target_answer, option_symbols_re, option_symbols)
    if match is not None:
        return match

    # try to match using the full text of the options
    match = get_first_match(target_answer, option_values, option_symbols)
    if match is not None:
        return match

    return None

###########
# Following code from SQUAD, here:
# https://github.com/huggingface/transformers/blob/main/src/transformers/data/metrics/squad_metrics.py

def normalize_answer(s):
    """Removing articles and punctuation, and standardizing whitespace are all typical text processing steps."""
    import string, re

    def remove_articles(text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()

def compute_exact(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))

def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

############

def evaluate_hf_model_qa(model: AutoModelForCausalLM, 
                         tokenizer: AutoTokenizer, 
                         data: Iterable,
                         start_prompt: str='### Consider the following question with context: ',
                         end_prompt: str=' ### Please answer with one of the options listed in the brackets:',
                         question_column: str='input',
                         answer_column: str='output',
                         max_samples: int=None,
                         min_new_tokens: int=0,
                         max_new_tokens: int=50,
                         remove_suffix: str=None,
                         device: str='cuda') -> dict:
    """
    Evaluate a Hugging Face model on a QA task.
    """
    exact_match, f1 = [], []
    model.to(device)  # Ensure the model is on the correct device

    for idx in tqdm(range(min(max_samples, len(data))), desc='Evaluating QA model'):
        question = data[idx][question_column]
        ground_truth = data[idx][answer_column]

        # Generate and decode the output string, removing the special tokens and any suffixes
        decoded = generate_from_prompt(model=model, 
                                       tokenizer=tokenizer, 
                                       input_data=question, 
                                       start_prompt=start_prompt, 
                                       end_prompt=end_prompt, 
                                       min_new_tokens=min_new_tokens,
                                       max_new_tokens=max_new_tokens)

        # Remove the suffix if specified - note that Mistral-Instruct models add a </s> suffix to specify the end of the output
        if remove_suffix is not None:
            decoded = decoded.replace(remove_suffix, '')

        exact_match.append(compute_exact(decoded, ground_truth))
        f1.append(compute_f1(decoded, ground_truth))

    return {
        'exact_match': np.mean(exact_match),
        'squad_f1_score': np.mean(f1)
    }


def evaluate_openai_model_qa(bot: DialogueBot,
                             data: Iterable, 
                             question_column: str,
                             answer_column: str,
                             max_samples: int=None) -> dict:
    """
    Evaluate an OpenAI model on a dataset using QA metrics.
    """

    exact_match, f1 = [], []

    # Iterate over the dataset
    for idx in tqdm(range(min(max_samples, len(data))), desc='Evaluating OpenAI QA model'):

        # Create the input string, framing it as a question
        input = f"Question: {data[idx][question_column]}\nAnswer:"
        
        # Get the model's response, which is the generated answer
        output = bot.return_bot_response(input)
        
        # Compute metrics
        exact_match.append(exact_match_score(output, data[idx][answer_column]))
        f1.append(f1_score(output, data[idx][answer_column]))

    return {
        'exact_match': np.mean(exact_match),
        'squad_f1_score': np.mean(f1),
    }

if __name__ == '__main__':

    # Parse the command line arguments
    parser = argparse.ArgumentParser(description='Evaluate a model on a QA task.')

    # Model arguments
    parser.add_argument('--model_type', type=str, help='The type of model to evaluate (Huggingface or OpenAI)', default='hf')
    parser.add_argument('--hf_model_id', type=str, help='The Huggingface model to evaluate', default='llama-2-7b-chat-hf')
    parser.add_argument('--oai_model_id', type=str, help='The OpenAI model ID to use', default='gpt-3.5-turbo')

    # Dataset arguments
    parser.add_argument('--dataset', type=str, help='The dataset to evaluate on', default='lurosenb/medqa')
    parser.add_argument('--dataset_revision', type=str, help='The revision of the dataset to use', default='latest')
    parser.add_argument('--split', type=str, help='The split of the dataset to evaluate on', default='test')
    parser.add_argument('--question_column', type=str, help='The name of the question column in the dataset', default='input')
    parser.add_argument('--answer_column', type=str, help='The name of the answer column in the dataset', default='output')
    parser.add_argument('--max_samples', type=int, help='The maximum number of samples to evaluate', default=200)

    # Generation arguments
    parser.add_argument('--max_tokens', type=int, help='The maximum number of tokens to generate', default=50)
    parser.add_argument('--remove_suffix', type=str, help='The suffix to remove from the generated output', default=None)

    # Environment and reproducibility arguments
    parser.add_argument('--device', type=str, help='The device to use for inference', default='cuda')
    parser.add_argument('--seed', type=int, help='The random seed to use', default=42)
    parser.add_argument('--save_dir', type=str, help='The directory to save the results to', default='results')

    # W&B logging arguments
    parser.add_argument('--wandb_logging', type=str, default='False', help='Whether to log to W&B.')
    parser.add_argument('--wandb_name', type=str, default='qa_eval', help='The name of the W&B project, for logging.')
    parser.add_argument('--wandb_api_var', type=str, default='WANDB_API_KEY', help='Name of the WandB API key variable name.')

    # Parse the arguments
    args = parser.parse_args()

    # Set the random seed for reproducibility
    torch.manual_seed(args.seed)

    # Initialize W&B
    if args.wandb_logging == 'True':
        wandb.login(key=getenv(args.wandb_api_var))
        wandb.init(project=args.wandb_name, 
                   name=args.run_name, 
                   config=args)
    
    # Load the test split of the dataset
    print('Loading dataset: ', args.dataset)
    data = load_dataset(args.dataset, args.dataset_revision, split=args.split)

    # Model evaluation logic based on the model type
    if args.model_type == 'hf':
        # Load the Hugging Face model and tokenizer
        print('Loading Hugging Face model: ', args.hf_model_id)
        tokenizer = AutoTokenizer.from_pretrained(args.hf_model_id)
        model = AutoModelForCausalLM.from_pretrained(args.hf_model_id).to(args.device)
        model.eval()

        # Evaluate the Hugging Face model
        print('Evaluating Hugging Face model on QA task: ', args.hf_model_id)
        qa_metrics = evaluate_hf_model_qa(model, tokenizer, data, args.question_column, args.answer_column, args.max_samples)

    elif args.model_type == 'openai':
        # NOTE: OpenAI Diaglogue bot QandA task has not been tested
        # TODO: Test
        # Evaluate the OpenAI model
        print('Evaluating OpenAI model on QA task: ', args.oai_model_id)
        bot = DialogueBot(model=args.oai_model_id, system_prompt=args.system_prompt)
        qa_metrics = evaluate_hf_model_qa(model, tokenizer, data, args.question_column, args.answer_column, args.max_samples, args.device)

    else:
        raise ValueError('Invalid model type: ', args.model_type)

    # Print the metrics to the console
    print('Model QA Metrics:')
    for key, value in qa_metrics.items():
        print(f'{key}: {value}')

    # Add the model and dataset names to the metrics dictionary
    metrics = {**vars(args), **qa_metrics}

    # Save the metrics to a JSON file
    model_id = args.hf_model_id if args.model_type == 'hf' else args.oai_model_id
    save_path = path.join(args.save_dir, f'{model_id.replace("/", "-")}_qa_metrics.json')
    print('Saving QA metrics to: ', save_path)

    if not path.exists(args.save_dir):
        makedirs(args.save_dir)

    with open(save_path, 'w') as f:
        json.dump(metrics, f)

    # Log the metrics to W&B
    if args.wandb_logging == 'True':
        wandb.log(metrics)
        wandb.finish()
