import openai
import time
import pandas as pd
import subprocess
import re
import os
import sys
import yaml
from tqdm import tqdm
from itertools import product
from codex_api_key import API_KEY
import test_gen_util

def get_completion(prompt, assrt, max_tokens=100, temperature=1.0,retry=False):
    if assrt:
        function = f"{prompt}\n    ?\n{assrt}"
    else:
        function = prompt
    messages = [
        {"role": "user", "content": function},
        {
            "role": "user",
            "content": "Replace the question mark in the above ocaml function with ocaml code that passes the assert.",
        },
    ]

    jsonobj = {
        "model": "gpt-3.5-turbo",
        # "prefix": "(* ocaml code *)",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    try:
        resp = openai.ChatCompletion.create(**jsonobj)
    except openai.OpenAIError: 
        if not retry: 
            print('retrying')
            time.sleep(10)
            return get_completion(prompt,assrt=assrt,max_tokens=max_tokens,temperature=temperature,retry=True)
        return (None,{})

    insertion = (
        resp["choices"][0]["message"]["content"] if "choices" in resp.keys() else None
    )

    if insertion:
        # print(f'completion:"{insertion}"')
        return insertion, resp
    return (None, resp)


def get_edit_completion(prompt, assert_=None, max_tokens=60, temperature=1.0,retry=False):
    if assert_:
        function = f"{prompt}\n    ?\n{assert_}"
    else:
        function = prompt
    
    # print(f'input is:"{function}"')

    instruction = "Replace the question mark in the above ocaml function with ocaml code that passes the assert."
    model = "code-davinci-edit-001"
    try: 
        resp = openai.Edit.create(
            model=model, input=function, instruction=instruction, temperature=temperature
        )
    except openai.OpenAIError: 
        if not retry: 
            print('retrying')
            time.sleep(10)
            return get_edit_completion(prompt,assert_=assert_,max_tokens=max_tokens,temperature=temperature,retry=True)
        return (None,{})

    insertion = resp["choices"][0]["text"] if "choices" in resp.keys() else None
    if insertion:
        # print(f'edit_comp:"{insertion}"')
        return insertion, resp
    return (None, resp)

def get_code_edit(prompt, assert_=None, max_tokens=60, temperature=1.0,retry=True):
    if assert_:
        function = f"{prompt}\n    ?\n{assert_}"
    else:
        function = prompt

    instruction = "Edit the function function to produce ocaml code that passes the assert."
    model = "code-davinci-edit-001"

    try:
        resp = openai.Edit.create(
            model=model, input=function, instruction=instruction, temperature=temperature
        )
    except openai.OpenAIError: 
        if not retry: 
            print('retrying')
            time.sleep(10)
            return get_code_edit(prompt,assert_=assert_,max_tokens=max_tokens,temperature=temperature,retry=True)
        return (None,{})
    insertion = resp["choices"][0]["text"] if "choices" in resp.keys() else None
    if insertion:
        # print(f'edit_:"{insertion}"')
        return insertion, resp
    return (None, resp)



def test_input(text, assert_, use_temp_file="temp.txt"):
    test_text = text + "\n" + assert_
    # print(test_text)
    with open(use_temp_file, "w") as file:
        file.write(test_text)
    prc = subprocess.run(
        ["ocaml", "temp.txt"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    return prc.returncode == 0


def make_log(test, response, accurate, mode, test_n, test_name, failed=False):
    log_json = {
        "test_n": test_n,
        "test": test,
        "accurate": accurate,
        "mode": mode,
        "failed": failed,
        "response": response,
        "test": test_name,
    }
    return pd.Series(log_json)


# read config file
def read_params(config_path: str):
    with open(config_path, "r") as file:
        params = yaml.safe_load(file)

    # can speicify to load auto-generated config file for train and test envs
    for name in ["env", "eval"]:
        if (
            name in params.keys()
            and len(params[name]) == 1
            and "get_from" in params[name]
        ):
            params_file =  params[name]["get_from"]
            with open(params_file, "r") as env_file:
                env_params = yaml.safe_load(env_file)
            params[name] = env_params[name]  # allows us to define both in same file
            print(f"fetched {name} params from file {params_file}")
            print(env_params[name])

    return params


def get_num_files(
    raw: bool, get_template: bool,params, base_dir=".",  fold="env"
):
    params 
    fold_params = params[fold]
    assn_dir = os.path.join(base_dir, fold_params["assignment_dir"])
    if get_template:
        assert os.path.isdir(temp_dir := os.path.join(assn_dir, "templates"))
        num_files = sum(
            bool(file_n.is_file() and re.match(r"\d+\.ml", file_n.name))
            for file_n in os.scandir(temp_dir)
        )
        return num_files, fold_params["assignment_dir"].split("/")[-2]
    else:
        # iterate through each assignment
        return (
            sum(fold_params["code_per_assignment"]),
            fold_params["assignment_dir"].split("/")[-2],
        )


def read_assns_from_params(
    raw: bool, get_template: bool,params, base_dir=".", fold="env"
):
    fold_params = params[fold]
    assn_dir = os.path.join(base_dir, fold_params["assignment_dir"])
    if get_template:
        assert os.path.isdir(temp_dir := os.path.join(assn_dir, "templates"))
        # try getting templates...
        for file_n in os.scandir(temp_dir):
            if file_n.is_file() and re.match(r"\d+\.ml", file_n.name):
                with open(file_n.path, "r") as file:
                    assn = file.read()
                if raw:
                    assn = assn.split("=")[0] + "=\n\t?\n" +'in\n'+ assn.split("in")[-1]
                yield assn, file_n

    else:
        # iterate through each assignment
        for assn_num, num_files in enumerate(fold_params["code_per_assignment"]):
            for fn in range(num_files):
                with open(os.path.join(assn_dir, assn_num, f"{fn}.ml"), "r") as file:
                    assn = file.read()
                if raw:
                    assn = assn.split("=")[0] + "\n\t?\n" + assn.split("in")[-1]
                yield assn, os.path.join(assn_dir, assn_num, f"{fn}.ml")


def run_codex_experiment(
    raw: bool,
    templates: bool,
    mode_name:str,
    params,
    run_codex_func = get_edit_completion, 
    fold: str = "env",
    n_samples:int = 20,
    min_time_per_cycle=None,
):
    counts = 0
    corrects = 0
    logs = []
    begin_time = time.time()

    n_tests, dirname = get_num_files(raw, templates, params, fold=fold)
    for (test, file_n),_ in (
        bar := tqdm(product(read_assns_from_params(raw, templates,params, fold=fold),range(n_samples)), total=n_tests*n_samples)
    ):
        # print(test,file_n)
        # the other (non-depricated endpoint) handles rate limits. This one does not.
        # Therefore we need to handle rate limits ourselves.
        # handle rate limit stuff (albeit jankily)
        end_time = time.time()
        if min_time_per_cycle is not None and (elapsed_time := end_time - begin_time) < min_time_per_cycle:
            # enforce min of two seconds per cycle to avoid rate limit
            time.sleep(min_time_per_cycle - elapsed_time)
        begin_time = time.time()
        # process request
        test = test.replace("!", "not")
        try: 
            res,json  = run_codex_func(test, None, max_tokens=100)
            assert_stmt = test.split("?")[1]
            func_body = res.split("in")[0]

            # check correctness and update counts
            correct = test_input(func_body, assert_stmt)
            if correct:
                corrects += 1
            counts += 1
            logs.append(make_log(test, func_body, correct, mode_name, file_n, dirname))
        except openai.InvalidRequestError: 
            logs.append(
                make_log(test, None, False, mode_name, file_n, dirname, failed=True)
            )
            print(f'failed:')


        success_rate = corrects / counts if counts != 0 else 0.0
        bar.set_description(f"success_rate = {success_rate:1.2f}")
        bar.bar_format = "{l_bar}{bar}{r_bar}" + f"file={file_n.name}"

    return success_rate, logs




def main(
    n_samples=5,
    raw=True,
    templates=True,
    fold="env",
):
    openai.api_key = API_KEY
    params = read_params('params.yaml')
    logs = []

    # do gpt 3.5 / 'completion' task
    n_tests, dirname = get_num_files(raw, templates,params, fold=fold)

    chat_success_rate, completion_logs = run_codex_experiment(
        raw=raw,
        templates=templates,
        params=params,
        mode_name = 'chat',
        run_codex_func=get_completion,
        fold=fold,n_samples=n_samples,
        min_time_per_cycle=None)
    
    logs.append(completion_logs)

    print(f"Chat completion format score for '{dirname}' = {chat_success_rate}")

    edit_success_rate, edit_logs = run_codex_experiment(
        raw=raw,
        templates=templates,
        params=params,
        mode_name = 'edit',
        run_codex_func=get_edit_completion,
        fold=fold,n_samples=n_samples,min_time_per_cycle=3.001)

    logs.append(edit_logs)

    print(f"edit completion format score for '{dirname}' = {edit_success_rate}")

    with open("baselines/data/coarse_logs.txt", "a") as file:
        file.write("mode, accuracy,test_set\n")
        file.write(f"chat, {chat_success_rate}, {dirname}\n")
        file.write(f"edit, {edit_success_rate}, {dirname}\n")

    fine_logs = pd.DataFrame(logs)
    print(fine_logs)
    with open(f"baselines/data/fine_logs_{dirname}.csv", "w") as file:
        fine_logs.to_csv(file)
    with open("baselines/data/fine_logs_{dirname}.json", "w") as file:
        fine_logs.to_json(file)
    with open("baselines/data/fine_logs_{dirname}.pikl", "wb") as file:
        fine_logs.to_pickle(file)


if __name__ == "__main__":
    test_gen_util.generate_tests('params.yaml')
    main()
