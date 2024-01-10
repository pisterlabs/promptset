import datetime
import glob
import os
import shutil
import subprocess
import time
import uuid
from typing import (Any, Callable, Dict, Generic, List, Optional, Set, Tuple,
                    Type, TypeVar, Union, cast, overload)

import openai
import pandas as pd
import tiktoken

from config import OPENAI_TOKEN, PROMPT_LIST, STOP_PROMPT
from utils import *

openai.api_key = OPENAI_TOKEN


# ask ChatGPT (Now it's official)
def call_evaluate_loop(prompt: str, model_name: str = "gpt-4", max_call: int = 5, prompt_ver: str = "v2"):
    messages = [
        {"role": "system", "content": "You are a helpful assistant who is an expert in biology, computer science, and engineering."},
        {"role": "user", "content": prompt},
    ]
    _uuid = uuid.uuid4()
    for i in range(max_call):
        print(f"call evaluate loop {i+1}/{max_call}")
        print(f"asking ChatGPT... model: {model_name}")
        print(f"messages: {messages}")
        chat_response = openai.ChatCompletion.create(model=model_name, messages=messages)
        print(f"\n***************** chat_response:\n{chat_response.choices[0]['message']['content']}")
        filepath = save_prompt_and_answer_with_modelname(
            f"chat_loop_{i+1}_{_uuid}", messages[-1]["content"], chat_response.choices[0]["message"]["content"], model_name, 0.9, 100, prompt_ver=prompt_ver
        )
        print(f"saved to {filepath}")
        script_path = prepare_python_script(filepath)

        if script_path is None:
            print(f"No Python script found.")
            break

        print(f"saved to {script_path}")
        _, output_text, result_type = run_opentrons_simulate(script_path, "eva_loop.txt")
        if result_type == "error":
            print(f"error occured. {output_text}")
            messages.append(
                {
                    "role": "user",
                    "content": f"I got this error:\n```\n{output_text}\n```\ncan you fix it? Make sure you only answer Python script.",
                }
            )
            continue
        elif result_type == "ok":
            # if this is the final conversation, rename the filepath to the last one, and break the loop
            os.rename(filepath, filepath.replace(f"chat_loop_{i+1}_{_uuid}", f"chat_loop_{i+1}_last_{_uuid}"))
            print(f"success! {output_text}")
            break

    return


def call_gpt3_4_model(prompt: str, model_name: str = "gpt-3.5-turbo", prompt_ver: str = "v2") -> str:
    # ask ChatGPT (Now it's official)
    chat_response = openai.ChatCompletion.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant who is an expert in biology, computer science, and engineering."},
            {"role": "user", "content": prompt},
        ],
    )

    _ = save_prompt_and_answer_with_modelname(f"single", prompt, chat_response.choices[0]["message"]["content"], model_name, 0.9, 100, prompt_ver=prompt_ver)
    return chat_response.choices[0]["message"]["content"]


def call_weak_model(prompt: str, model_name: str = "davinci", prompt_ver: str = "v2") -> None:
    # ask ChatGPT (Now it's official)
    res = openai.Completion.create(
        model=model_name,
        prompt=prompt,
        max_tokens=1600 if model_name == "text-ada-001" else 2500,
        temperature=0.8,
        n=1,
    )

    filepath = save_prompt_and_answer_with_modelname(f"single", prompt, res.choices[0]["text"], model_name, 0.9, 100, prompt_ver=prompt_ver)
    return None


model_call_list = {
    # "text-ada-001": call_weak_model, # call once
    # "text-davinci-003": call_weak_model, # call once
    # "gpt-3.5-turbo": call_gpt3_4_model, # call once
    # "gpt-4": call_gpt3_4_model, # call once

    "gpt-3.5-turbo": call_evaluate_loop,  # chat loop
    "gpt-4": call_evaluate_loop,  # chat loop
}

def main(prompt_version="v1"):
    prompt = PROMPT_LIST[prompt_version]
    for model_name, model_call in model_call_list.items():
        print(f"############### call model {model_name}, prompt version {prompt_version} ###############")
        try:
            model_call(prompt, model_name, prompt_ver=prompt_version)
        except Exception as e:
            time.sleep(5)
            import traceback

            print(f"error occured. {e}")
            traceback.print_exc()
            model_call(prompt, model_name, prompt_ver=prompt_version)
            continue
    pass


if __name__ == "__main__":
    # prompt_versions = ["v2.cell2", "v2.medium2", "v2.cell2.medium2", "v2-B", "v2-B.cell2", "v2-B.target2", "v2-B.cell2.target2"]
    prompt_versions = ["v3"]
    n_calls = 2

    for prompt_version in prompt_versions:
        print(f"********** prompt version {prompt_version} **********")
        for i in range(n_calls):
            print(f"\n%%% run {i} times %%%\n")
            try:
                main(prompt_version=prompt_version)
                time.sleep(5)
            except Exception as e:
                print(f"error occured. {e}")
                time.sleep(5)
                continue
