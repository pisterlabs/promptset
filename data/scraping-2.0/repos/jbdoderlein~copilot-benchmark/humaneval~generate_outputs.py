"""
Generate outputs for HumanEval based prompts using Codex and Copilot
"""

import argparse
import json
import os
from time import sleep
from pathlib import Path

import openai

import copilot

with open('config.json', 'r') as f:
    config = json.load(f)

def replace_original_fname(name, content):
    """
    Replace the function name of humaneval context. Usefull for signature_name modification
    """
    lines = content.split('\n')
    def_lines = [line for line in lines if line.startswith("def")]
    line = def_lines[-2]
    fname = line.split("def")[1].split("(")[0].strip()
    if fname == name:
        return content
    else:
        return content.replace(fname, name)

if '__main__' in __name__:
    parser = argparse.ArgumentParser(description='Generate solution from copilot/codex and test')
    parser.add_argument('directory', type=str,
                        help='Sample directory')
    parser.add_argument('-k', '--key', type=int, help="Index of key", default=0)
    parser.add_argument('-e', '--engine', type=str, nargs='+',
                            default=['copilot', 'codex', 'codex_100'])
    parser.add_argument('-t', '--temperature', type=float, nargs='+', default=[0.0,0.2,0.4,0.6,0.8,1.0])

    args = parser.parse_args()
    lib_path = args.directory
    engines = args.engine
    temperatures = args.temperature

    copilot_instance = copilot.Copilot(
        agent_path=os.path.join(os.path.dirname(copilot.__file__), 'dist','agent.js')
    )
    openai.api_key = config["OPENAI_API_KEYS"][args.key]

    PROMPT_PATH = "prompts"
    OUTPUT_PATH = "outputs"
    TEST_PATH = "tests"
    RESULT_PATH = "results"

    CODEX_SLEEP_COOLDOWN = 20

    input_path = os.path.join(PROMPT_PATH, lib_path)
    for temperature in temperatures:
        for engine in engines:
            Path(
                os.path.join(OUTPUT_PATH, str(temperature), engine,  lib_path)
            ).mkdir(parents=True, exist_ok=True)

    for i, f in enumerate(os.listdir(input_path)):
        print(f"({i+1}/{len(os.listdir(input_path))}) : {f}", flush=True)
        if os.path.isfile(os.path.join(input_path, f)):
            for temperature in temperatures:
                print(f"Temperature : {temperature}", flush=True)
                for engine in engines:
                    final_path = os.path.join(OUTPUT_PATH, str(temperature), engine,  lib_path)
                    match engine:
                        case "copilot":
                            print("-> Copilot ", end='', flush=True)
                            if temperature == 1.0 and (not os.path.isfile(os.path.join(final_path ,f))): # Define copilto as temp 1
                                document = copilot.Document(os.path.join(input_path, f), "python")
                                completions = copilot_instance.get_all_completion(document)
                                with open(os.path.join(TEST_PATH, f), 'r') as f2:
                                    test = f2.read()
                                    completions.content += test
                                if lib_path == "signature_name" or lib_path == "signature_full":
                                    completions.content = replace_original_fname(f.replace('.py', ''), completions.content)
                                completions.save(os.path.join(final_path,f))
                                print(" Done", flush=True)
                            else:
                                print(" Skipped", flush=True)
                        case "codex":
                            print("-> Codex ", end='', flush=True)
                            if not os.path.isfile(os.path.join(final_path ,f)):
                                with open(os.path.join(input_path, f),'r') as f2:
                                    content_prefix, content_suffix = f2.read().split("$$$")

                                codex_success = False
                                while not codex_success:
                                    try:
                                        api_result = openai.Completion.create(
                                            model="code-davinci-002",
                                            prompt=content_prefix,
                                            suffix=content_suffix,
                                            temperature=temperature,
                                            max_tokens=256
                                        )
                                        codex_success = True
                                    except openai.error.RateLimitError:
                                        print(" | API Cooldown", end="", flush=True)
                                        sleep(CODEX_SLEEP_COOLDOWN)
                                if len(api_result["choices"]) > 0:
                                    file_content = content_prefix+api_result["choices"][0]["text"]+content_suffix

                                with open(os.path.join(TEST_PATH, f), 'r') as f2:
                                    test = f2.read()
                                    file_content += test

                                if lib_path == "signature_name" or lib_path == "signature_full":
                                    file_content = replace_original_fname(f.replace('.py', ''), file_content)
                                with open(os.path.join(final_path , f), 'w') as f2:
                                    f2.write(file_content)
                                print(" Done", flush=True)
                            else:
                                print(" Skipped", flush=True)
                        case "codex_10"|"codex_100":
                            Path(os.path.join(final_path ,f.replace(".py", ""))).mkdir(parents=True, exist_ok=True)
                            n = int(engine.split("_")[-1])
                            n_al = max([int(file.split('.py')[0]) for file in os.listdir(os.path.join(final_path, f.replace('.py', '')))]+[0])
                            print(f"-> Codex {n} ({n_al}/{n} already generated) ", end='', flush=True)
                            if n_al < n:
                                with open(os.path.join(input_path, f),'r') as f2:
                                    content_prefix, content_suffix = f2.read().split("$$$")

                                codex_success = False
                                while not codex_success:
                                    try:
                                        api_result = openai.Completion.create(
                                            model="code-davinci-002",
                                            prompt=content_prefix,
                                            suffix=content_suffix,
                                            temperature=temperature,
                                            max_tokens=256,
                                            n=n-n_al
                                        )
                                        codex_success = True
                                    except openai.error.RateLimitError:
                                        print(" | API Cooldown", end="", flush=True)
                                        sleep(CODEX_SLEEP_COOLDOWN)
                                
                                with open(os.path.join(TEST_PATH, f), 'r') as f2:
                                    test_content = f2.read()

                                for i, choice in enumerate(api_result["choices"]):
                                    with open(os.path.join(final_path ,f.replace(".py", ""), f"{n_al+i+1}.py"), 'w') as f2:
                                        file_content = content_prefix+choice["text"]+content_suffix+"\n\n"+test_content
                                        file_content = replace_original_fname(f.replace('.py', ''), file_content)
                                        f2.write(file_content)
                                print(" Done", flush=True)
                            else:
                                print(" Skipped", flush=True)
