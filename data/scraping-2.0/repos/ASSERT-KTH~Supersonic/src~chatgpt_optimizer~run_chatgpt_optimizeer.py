#!/usr/bin/env python
# -*- encoding: utf-8 -*-


'''
@Author  :   Sen Fang
@Email   :   senf@kth.se
@Ide     :   vscode & conda
@File    :   run_chatgpt_optimizeer.py
@Time    :   2023/03/25 18:43:57
'''


"""In this file, we use the chatgpt API generate the optimized code for each original c or cpp file from prompt file."""


import os
import argparse
import re
import time
import random
import itertools
import sys
import json
import string
import subprocess
import logging
import tempfile
import openai
import tiktoken

from tqdm import tqdm
from pathlib import Path
from typing import Optional, Tuple


SLEEP_TIME = 60


def load_prompts(dir_path: str):
    """Load prompt files from submission-easy directory."""
    paths = []
    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.startswith("original.c") or file.startswith("original.cpp"):
                path = os.path.join(root, file)
                paths.append(path)
    return paths


def extract_code(response: str):
    """Extract the optimized code from the response."""
    lines = response.split("\n")
    code = []
    cnt = 0
    for line in lines:
        if line.startswith("```"):
            if cnt == 0:
                cnt += 1
                continue
            else:
                break
        code.append(line)
    return "\n".join(code)


def write_response(path: str, response_dict: dict):
    """Write the response into response_{}.c or response_{}.cpp file."""
    files = os.listdir(path.replace(path.split("/")[-1], ""))
    original_file = "original_{}.txt"
    if "original.c" in files:
        response_file = "response_{}.c"
    elif "original.cpp" in files:
        response_file = "response_{}.cpp"
    else:
        raise ValueError("No original file found.")
    for key, value in response_dict.items():
        with open(path.replace(path.split("/")[-1], original_file.format(key)), "w") as f:
            f.write(value[0])
        with open(path.replace(path.split("/")[-1], response_file.format(key)), "w") as f:
            f.write(value[1])


def generate_optimized_code(path: str, prompt: str, model_name: str="gpt-3.5-turbo", temperature: float=0.7, max_tokens: int=2048, num_requests: int=10):
    """Feed prompt files into ChatGPT and save the generated optimized code into submission-easy directory."""

    with open(path, "r") as f:
        original_code = f.read()
    
    # build markdown code block
    original_code = "```\n" + original_code + "\n```"

    input_length = len(tiktoken.encoding_for_model(model_name).encode(original_code))

    if input_length > max_tokens:
        with open(path.replace(path.split("/")[-1], "response.txt"), "a") as f:
            f.write("The length of prompt is too long.")
        return
    
    response_dict = {}
    for i in range(num_requests):
        response_dict[str(i)] = []
        response = openai.ChatCompletion.create(
            model=model_name,
            messages=[{"role": "system", "content": prompt},
                      {"role": "user", "content": original_code}],
            temperature=temperature,
        )
        # print(response)
        original_text = response["choices"][0]["message"]["content"]
        response_dict[str(i)].append(original_text)
        optimized_code = extract_code(response["choices"][0]["message"]["content"])
        response_dict[str(i)].append(optimized_code)

    return response_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_level", "-p", type=str, default="easy", help="The level of prompt to use.", choices=["easy", "specific", "advanced"])
    parser.add_argument("--model_name", "-mn", type=str, default="gpt-3.5-turbo", help="The name of the model to use.")
    parser.add_argument("--only_generate", "-og", type=bool, default=True, help="Only generate the optimized code or not.")
    parser.add_argument("--only_test", "-ot", type=bool, default=False, help="Only test the generated optimized code or not.")
    parser.add_argument("--num_test", "-nt", type=int, default=None, help="Test the script with a limited number of prompts.")
    parser.add_argument("--num_requests", "-nr", type=int, default=10, help="The number of requests to send for each prompt.")
    parser.add_argument("--temperature", "-t", type=float, default=0.7, help="The temperature of the model.") # The default value of web chatgpt is 0.7
    parser.add_argument("--max_tokens", "-mt", type=int, default=3000, help="The maximum number of tokens to generate.")

    args = parser.parse_args()

    # Set API key
    openai.api_key = os.getenv("OPENAI_API_KEY")

    # Load prompt files
    dir_name = "submissions" + "-" + args.prompt_level
    paths = load_prompts(dir_name)

    prompt_dict = {"easy": "Please optimize the following code and return the markdown format optimized code.\n",
                   "advanced": "I want you to act as an experienced C and C++ developer and your task is to optimize my written C or C++ programs. I want you to optimize my program from the running time and memory usage. I will type my C or C++ program and you will optimize the program and return the optimized program. I want you to only reply with the fixed program inside one unique code block, and nothing else. Do not write explanations unless I instruct you to do so.\n"}
    prompt = prompt_dict[args.prompt_level]

    for idx, path in tqdm(enumerate(paths)):

        output_file = "response" + "_" + str(args.prompt_level) + "_" + str(args.temperature) + ".json"
        if args.only_generate:
            # print(idx)s
            
            if idx < 188:
                continue
            print(path)
            try:
                response_dict = generate_optimized_code(path, prompt=prompt, model_name=args.model_name, temperature=args.temperature, max_tokens=args.max_tokens, num_requests=args.num_requests)
            except Exception as e:
                print(e)
                print(idx)
                break
            
            if response_dict is not None:
                write_response(path, response_dict)
            time.sleep(SLEEP_TIME)
        elif args.only_test:
            pass
            # with open(path.replace("prompt.txt", output_file), "r") as f:
            #     response_dict = json.load(f)
            # for key, value in response_dict.items():
            #     files = os.listdir(path.replace("prompt.txt", ""))
        else:
            pass

        if args.num_test is not None:
            if idx >= args.num_test:
                break

