import argparse
import openai
import time
from openai.error import OpenAIError
import os
from tqdm import tqdm
import json
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pathlib
import uuid
import sys
import re

sys.path.append("..")

from utils import read_json, write_json, generate_unique_id, MODEL_COSTS

#openai.api_key = args.os.getenv("OPENAI_API_KEY")

CHAT_COMPLETION_MODELS = ["gpt-3.5-turbo", "gpt-4"]
TEXT_COMPLETION_MODELS = ["text-davinci-003"]

def chat_completion(messages, model="gpt-3.5-turbo", return_text=True, return_usage=True, model_args=None):
    if model_args is None:
        model_args = {}

    while True:
        try:
            response = openai.ChatCompletion.create(model=model, messages=messages, **model_args)
            text = response["choices"][0]["message"]["content"].strip()
            usage = response["usage"]
            
            if return_text and return_usage:
                return text, usage
            
            if return_text:
                return text
            
            if return_usage:
                return usage

            return response
        except OpenAIError as e:
            print("OpenAI error. Waiting for 1 minute.")
            time.sleep(60)
            continue

def text_completion(prompt, model="text-davinci-003", return_text=True, return_usage=True, model_args=None):
    if model_args is None:
        model_args = {}

    while True:
        try:
            response = openai.Completion.create(model=model, prompt=prompt, **model_args)
            text = response["choices"][0]["text"].strip()
            usage = response["usage"]

            if return_text and return_usage:
                return text, usage
            
            if return_text:
                return text
            
            if return_usage:
                return usage

            return response
        except OpenAIError as e:
            print("OpenAI error. Waiting for 1 minute.")
            time.sleep(60)
            continue

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, help="Path to evaluation data in json", required=True)
    parser.add_argument("--openai-key", type=str, help="OpenAI API Key", required=True)
    parser.add_argument("--model", type=str, help="Model to use for evaluation", default="gpt-3.5-turbo")
    parser.add_argument("--temperature", type=float, help="Temperature for generation", default=0.3)
    parser.add_argument("--max-tokens", type=int, help="Max tokens for generation", default=40)
    parser.add_argument("--top-p", type=float, help="Top p for generation", default=1)
    parser.add_argument("--frequency-penalty", type=float, help="Frequency penalty for generation", default=0)
    parser.add_argument("--presence-penalty", type=float, help="Presence penalty for generation", default=0)
    parser.add_argument("--output-dir", type=str, help="Output directory for evaluation results", default="outputs")
    parser.add_argument("--num-samples", type=int, help="Number of samples to evaluate", default=0)
    parser.add_argument("--ignore-path", type=str, help="Path to already evaluated data", default=None)
    
    args = parser.parse_args()
    openai.api_key = args.openai_key 
    data = read_json(args.datapath)
    
    ignore_map = {}

    if args.ignore_path is not None:
        ignore_data = read_json(args.ignore_path)
        
        for sample in ignore_data["data"]:
            ignore_map[sample["instance_id"]] = sample

    if args.num_samples > 0:
        data = data[:int(args.num_samples)]

    predictions = []
    references = []

    outputs = {
        "metadata": {
            "datapath": args.datapath,
            "model": args.model,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "top_p": args.top_p,
            "frequency_penalty": args.frequency_penalty,
            "presence_penalty": args.presence_penalty
        },
        "metrics": {
            "accuracy": 0,
            "precision": 0,
            "recall": 0,
            "f1": 0,
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            },
            "cost": {
                "input": 0,
                "output": 0,
                "total": 0
            }
        },
        "data": data
    }

    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    datapath = pathlib.Path(args.datapath)
    
    output_path = os.path.join(args.output_dir, f"{datapath.stem}_{args.model}_{generate_unique_id()}.json")
    print(f"Writing to {output_path}")
    
    for sample in tqdm(data, total=len(data)):
        if sample["instance_id"] in ignore_map:
            ignore_instance = ignore_map[sample["instance_id"]]
            if "response" in ignore_instance:
                sample.update(ignore_instance)
                continue
    
        if "response" in sample:
            continue

        if args.model in CHAT_COMPLETION_MODELS:
            response, usage = chat_completion([{"role": "user", "content": sample["prompt"].strip()}], model=args.model, return_text=True, return_usage=True, model_args={
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "top_p": args.top_p,
                "frequency_penalty": args.frequency_penalty,
                "presence_penalty": args.presence_penalty
            })
        elif args.model in TEXT_COMPLETION_MODELS:
            response, usage = text_completion(sample["prompt"].strip(), model=args.model, return_text=True, return_usage=True, model_args={
                "temperature": args.temperature,
                "max_tokens": args.max_tokens,
                "top_p": args.top_p,
                "frequency_penalty": args.frequency_penalty,
                "presence_penalty": args.presence_penalty
            })
        else:
            raise ValueError(f"Model {args.model} not supported for evaluation.")

        sample["response"] = response
        sample["usage"] = usage
        outputs["metrics"]["usage"]["prompt_tokens"] += usage["prompt_tokens"]
        outputs["metrics"]["usage"]["completion_tokens"] += usage["completion_tokens"]
        outputs["metrics"]["usage"]["total_tokens"] += usage["total_tokens"]

        if sample["type"] in ["bcq", "bcq_with_kg"]:
            ref = 1 if sample["answer"].strip().lower() == "yes" else 0
            pred = 1 if response.strip().lower() == "yes" else 0
            references.append(ref)
            predictions.append(pred)
            sample["correct"] = ref == pred
        elif sample["type"] in ["bcq_cot", "bcq_cot_with_kg"]:
            ref = 1 if sample["answer"].strip().lower() == "yes" else 0
            match = re.search("<Answer>(?P<pred>.*)</Answer>", response)
            pred = 0

            if match:
                pred = match["pred"].strip().lower()
                pred = 1 if pred == "yes" else 0
            
            references.append(ref)
            predictions.append(pred)
            sample["correct"] = ref == pred
        elif sample["type"] == "mcq":
            try:
                gold_answers = [int(a) for a in sample["answer"].split(",")]
                gpt_answers = [int(a) for a in response.split(",")]
                refs = [1 if i+1 in gold_answers else 0 for i in range(sample["num_options"])]
                preds = [1 if i+1 in gpt_answers else 0 for i in range(sample["num_options"])] 
                
                sample["references"] = refs
                sample["predictions"] = preds
                sample["accuracy"] = accuracy_score(refs, preds)
                sample["precision"] = precision_score(refs, preds, average="macro")
                sample["recall"] = recall_score(refs, preds, average="macro")
                sample["f1"] = f1_score(refs, preds, average="macro")
            except ValueError:
                continue 
        else:
            raise ValueError(f"Type {sample['type']} not supported for evaluation.")
        
        write_json(outputs, output_path)

    if predictions:
        outputs["metrics"]["accuracy"] = accuracy_score(references, predictions)
        outputs["metrics"]["precision"] = precision_score(references, predictions, average="macro")
        outputs["metrics"]["recall"] = recall_score(references, predictions, average="macro")
        outputs["metrics"]["f1"] = f1_score(references, predictions, average="macro")
    else:
        outputs["metrics"]["accuracy"] = np.mean([sample["accuracy"] for sample in data if "accuracy" in sample])
        outputs["metrics"]["precision"] = np.mean([sample["precision"] for sample in data if "precision" in sample])
        outputs["metrics"]["recall"] = np.mean([sample["recall"] for sample in data if "recall" in sample])
        outputs["metrics"]["f1"] = np.mean([sample["f1"] for sample in data if "f1" in sample])

    outputs["metrics"]["cost"]["input"] = outputs["metrics"]["usage"]["prompt_tokens"] * MODEL_COSTS[args.model]["input"]
    outputs["metrics"]["cost"]["output"] = outputs["metrics"]["usage"]["completion_tokens"] * MODEL_COSTS[args.model]["output"]
    outputs["metrics"]["cost"]["total"] = outputs["metrics"]["cost"]["input"] + outputs["metrics"]["cost"]["output"]

    write_json(outputs, output_path)

if __name__ == "__main__":
    main()
