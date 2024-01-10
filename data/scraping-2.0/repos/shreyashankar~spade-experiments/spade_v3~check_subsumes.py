"""
Module to check whether a function is subsumed by a set of functions.
"""
import difflib
import json
import os
import openai
import ast
import inspect
from itertools import permutations
import numpy as np
import asyncio
import logging
import pandas as pd

from litellm import acompletion, completion
from litellm import Router

from dotenv import load_dotenv

import ast

# import logging
import re

from typing import List, Tuple

import litellm

# from litellm import RateLimitManager

## init RateLimitManager
# gpt_4_handler = RateLimitManager(max_requests_per_minute=60, max_tokens_per_minute=9000)
# gpt_4_2_handler = RateLimitManager(
#     max_requests_per_minute=450, max_tokens_per_minute=79000
# )

litellm.success_callback = ["langfuse"]
litellm.failure_callback = ["langfuse"]

load_dotenv()

openai.api_type = "azure"
openai.api_base = os.getenv("AZURE_API_BASE")
openai.api_version = "2023-07-01-preview"
openai.api_key = os.getenv("AZURE_API_KEY")

model_list = [
    {
        "model_name": "azure/gpt-4-2",
        "litellm_params": {  # params for litellm completion/embedding call
            "model": "azure/gpt-4-2",
            "api_key": os.getenv("AZURE_API_KEY"),
            "api_version": os.getenv("AZURE_API_VERSION"),
            "api_base": os.getenv("AZURE_API_BASE"),
            "rpm": 450,
            "tpm": 79000,
        },
    },
    {  # list of model deployments
        "model_name": "azure/gpt-35-turbo",  # model alias
        "litellm_params": {  # params for litellm completion/embedding call
            "model": "azure/chatgpt-v-2",  # actual model name
            "api_key": os.getenv("AZURE_API_KEY"),
            "api_version": os.getenv("AZURE_API_VERSION"),
            "api_base": os.getenv("AZURE_API_BASE"),
        },
    },
]


class CodeCleaner(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        # Remove the function name but keep the body
        node.name = ""
        self.generic_visit(node)  # Visit children nodes
        return node

    def visit_AsyncFunctionDef(self, node):
        # Handle async functions in the same way
        node.name = ""
        self.generic_visit(node)
        return node

    def visit_Expr(self, node):
        if isinstance(node.value, ast.Str):  # Remove docstrings
            return None
        return node


def get_clean_ast(source_code):
    # Parse the source code into an AST
    tree = ast.parse(source_code)

    # Clean the AST by removing function names and docstrings
    cleaner = CodeCleaner()
    cleaned_tree = cleaner.visit(tree)

    # Optionally, you can convert the cleaned AST back to source code
    # clean_source_code = ast.unparse(cleaned_tree)

    return cleaned_tree


def get_clean_ast_dup(source_code):
    # Parse the source code into an AST
    tree = ast.parse(source_code)

    # Clean the AST by removing function names and docstrings
    cleaner = CodeCleaner()
    cleaned_tree = cleaner.visit(tree)

    # Optionally, you can convert the cleaned AST back to source code
    # clean_source_code = ast.unparse(cleaned_tree)

    return cleaned_tree


def does_func1_imply_func2(func1, func2):
    ast1 = get_clean_ast(func1)
    ast2 = get_clean_ast(func2)

    # Compare the cleaned ASTs
    return ast.dump(ast1) == ast.dump(ast2)


def may_subsume(function_1, function_2, funcs, func_order, M):
    """Returns 0 if function_1 definitely does not subsume function_2, 1 if function_1 may subsume function_2, and 2 if function_1 definitely subsumes function_2."""
    idx_1 = func_order[function_1]
    idx_2 = func_order[function_2]

    # If there is any example where function_1 is 1 and function_2 is 0, then function_1 definitely does not subsume function_2.
    for i in range(M.shape[0]):
        if M[i, idx_1] == 1 and M[i, idx_2] == 0:
            return 0

    # If the source code is the same, return True
    if does_func1_imply_func2(funcs[function_1], funcs[function_2]):
        return 2

    # If there is `ask_llm` in both functions, return false
    # ask_llm_count = ("ask_llm" in funcs[function_1]) + ("ask_llm" in funcs[function_2])
    # if ask_llm_count == 2:
    #     return 0

    return 1


def replace_newlines_in_strings(s):
    def replace(match):
        # Replace newlines in the matched string
        return match.group(0).replace("\n", "\\n")

    # Regular expression to find double-quoted strings
    return re.sub(r'"([^"\\]*(?:\\.[^"\\]*)*)"', replace, s)


async def function_subsumes(
    func_a: str,
    func_b: str,
    func_a_src: str,
    func_b_src: str,
    prompt_template: str,
    response: str,
) -> bool:
    messages = [
        {
            "content": f"You are an expert Python programmer and are helping me remove redundant assertions for my LLM pipeline. My pipeline prompt template is `{prompt_template}` and an example response={response}.",
            "role": "system",
        },
        {
            "content": f'Here is my first function:\n{func_a_src}\n\nHere is my second function: {func_b_src}\n\nDoes the first function imply or not imply the second function? In other words, is there an example such that function `{func_b}` returns False for while function `{func_a}` returns True? If both functions contain `ask_llm` calls to check for the same thing, your answer should be no (meaning the first function implies the second). Return your answer as a JSON within ```json ``` ticks with keys `answer` (yes or no) and `response` ("N/A" if your answer is no). Yes means the first does not imply the second, and no means the first implies the second.',
            "role": "user",
        },
    ]

    try:
        response = await acompletion(
            model="azure/gpt-4",
            messages=messages,
            fallbacks=["azure/gpt-35-turbo"],
        )

        reply = response["choices"][0]["message"]["content"]
        # Extract answer within JSON markers ```json\n ``` and convert to dict

        # Find the JSON part within the reply
        json_part = re.search(r"```json(.*?)\n```", reply, re.DOTALL)

        if json_part:
            json_string = json_part.group(1).strip()
            json_string = replace_newlines_in_strings(json_string)

            # Convert the JSON string to a dictionary
            try:
                json_answer = json.loads(json_string)
            except Exception as e:
                json_answer = eval(json_string)

            answer = json_answer["answer"]
            rationale = json_answer["response"]

            if "yes" in str(answer).lower():
                print(f"Found example response {rationale}")
                return False, rationale

            print(f"Found no example; {func_a} subsumes {func_b}")
            return True, rationale

        print(f"Cound not find JSON part in reply: {reply}")
        return False, ""

    except Exception as e:
        logging.error(e, exc_info=True)
        print(json_part.group(1).strip())
        return False, ""


async def evaluate_all_subsumes(
    M,
    func_executables,
    func_order,
    prompt_template,
    response,
):
    # Get all pairs of functions (order matters)
    funcs = {func.__name__: inspect.getsource(func) for func in func_executables}
    pairs = list(permutations(funcs.keys(), 2))
    K = np.zeros((len(func_order), len(func_order)))

    results = []
    requests = []
    counter = 0
    awaited_results = []
    for pair in pairs:
        run_pair = may_subsume(pair[0], pair[1], funcs, func_order, M)
        if run_pair == 0:
            K[func_order[pair[0]], func_order[pair[1]]] = 0
            continue
        elif run_pair == 2:
            K[func_order[pair[0]], func_order[pair[1]]] = 1
            continue

        # See if a subsumes b by transitivity
        # if there is some c such that a subsumes c and c subsumes b, then a subsumes b
        for c in func_order.keys():
            if (
                K[func_order[pair[0]], func_order[c]] == 1
                and K[func_order[c], func_order[pair[1]]] == 1
            ):
                print(f"Using transitivity based on {c}")
                K[func_order[pair[0]], func_order[pair[1]]] = 1
                break

        counter += 1

        task = function_subsumes(
            pair[0],
            pair[1],
            funcs[pair[0]],
            funcs[pair[1]],
            prompt_template,
            response,
        )
        results.append(task)
        requests.append(pair)

        if counter % 50 == 0:
            # Put a barrier here otherwise we'll run out of tokens
            # when executing everything asynchronously
            awaited_results.extend(
                await asyncio.gather(*results, return_exceptions=True)
            )
            results = []
            await asyncio.sleep(5)

    # Run all outstanding tasks
    awaited_results.extend(await asyncio.gather(*results, return_exceptions=True))

    for i, result in enumerate(awaited_results):
        pair = requests[i]
        if isinstance(result, Exception):
            K[func_order[pair[0]], func_order[pair[1]]] = 0
            continue
        K[func_order[pair[0]], func_order[pair[1]]] = 1 if result[0] else 0

    print(f"Ran {counter} pairs out of {len(pairs)} pairs.")

    return K


def collate_subsumption_results(M, func_executables, func_order, K):
    # Get all pairs of functions (order matters)
    funcs = {func.__name__: inspect.getsource(func) for func in func_executables}
    pairs = list(permutations(funcs.keys(), 2))

    results = []

    for pair in pairs:
        run_pair = may_subsume(pair[0], pair[1], funcs, func_order, M)

        if run_pair == 0:
            results.append(
                {
                    "func A": funcs[pair[0]],
                    "func B": funcs[pair[1]],
                    "A -> B": False,
                    "asked_LLM": False,
                }
            )
            continue
        elif run_pair == 2:
            results.append(
                {
                    "func A": funcs[pair[0]],
                    "func B": funcs[pair[1]],
                    "A -> B": True,
                    "asked_LLM": False,
                }
            )
            continue

        else:
            results.append(
                {
                    "func A": funcs[pair[0]],
                    "func B": funcs[pair[1]],
                    "A -> B": bool(K[func_order[pair[0]], func_order[pair[1]]]),
                    "asked_LLM": True,
                }
            )

    results_df = pd.DataFrame(results)
    num_true = results_df["A -> B"].sum()
    print(f"Found {num_true} subsumptions out of {len(results_df)} pairs.")
    return results_df


async def sample_subsumption_prompts_and_results(
    M,
    func_executables,
    func_order,
    K,
    prompt_template,
    response,
):
    # Get all pairs of functions (order matters)
    funcs = {func.__name__: inspect.getsource(func) for func in func_executables}
    pairs = list(permutations(funcs.keys(), 2))

    results = []

    for pair in pairs:
        run_pair = may_subsume(pair[0], pair[1], funcs, func_order, M)

        if run_pair == 0:
            continue
        elif run_pair == 2:
            continue

        else:
            results.append(
                {
                    "prompt_template": prompt_template,
                    "response": response,
                    "func_a": pair[0],
                    "func_b": pair[1],
                    "func_a_src": funcs[pair[0]],
                    "func_b_src": funcs[pair[1]],
                    "result": bool(K[func_order[pair[0]], func_order[pair[1]]]),
                }
            )

    results_df = pd.DataFrame(results)
    return results_df


async def extract_subsumptions_from_source_code(func_map) -> List[Tuple[str, str]]:
    """Extracts subsumptions from source code."""
    # Uses GPT-4 to generate subsumptions from source code
    assertion_blob = "\n\n".join(func_map.values())
    format_instruction = 'Please return your answer as a JSON list within ```json ``` ticks, where each element of the list is a tuple (A, B). If two functions A and B check for the same thing, make sure to include both tuples (A, B) and (B, A). For example, if I only had two functions `check_json` and `assert_json`, the answer should be:\n ```json\n[("check_json", "assert_json"), ("assert_json", "check_json")]\n```'

    messages = [
        {
            "content": f"You are an expert Python programmer and are helping me identify redundant assertion functions.",
            "role": "system",
        },
        {
            "content": f"Here are all the functions I have:\n\n{assertion_blob}\n\nBased on the code, please identify every pair of functions where one function implies the other. Note that function A might imply function B, but function B may not imply function A. If two functions A and B check for the same thing, then they both imply each other (i.e., A implies B and B implies A), so you should list both directions. Feel free to use the function names to decide if two functions check for the same thing.",
            "role": "user",
        },
    ]
    # Function A implies B if, for all inputs where B returns False, A also returns False.

    response = await acompletion(
        model="azure/gpt-4-2",
        messages=messages,
        fallbacks=["azure/gpt-35-turbo"],
    )

    reply = response["choices"][0]["message"]["content"]
    print(reply)

    # Run a second time to get the answer formatted
    response = await acompletion(
        model="azure/gpt-4-2",
        messages=messages
        + [
            {"role": "assistant", "content": reply},
            {"role": "user", "content": format_instruction},
        ],
        fallbacks=["azure/gpt-35-turbo"],
    )
    reply = response["choices"][0]["message"]["content"]
    print(reply)

    # Extract answer within JSON markers ```json\n ``` and convert to dict

    # Find the JSON part within the reply
    json_part = re.search(r"```json(.*?)\n```", reply, re.DOTALL)
    print(json_part.group(1).strip())

    # Extract pairs from the JSON part
    pairs = eval(json_part.group(1).strip())

    # Only include pairs that are in the function map
    confirmed_pairs = []
    for pair in pairs:
        if pair[0] in func_map.keys() and pair[1] in func_map.keys():
            confirmed_pairs.append(pair)
        else:
            print(
                f"Skipping pair {pair} because at least one of the functions are hallucinated"
            )

    return confirmed_pairs


async def identify_subsumption_pairs(
    M,
    func_executables,
    func_order,
):
    # Get all pairs of functions (order matters)
    funcs = {func.__name__: inspect.getsource(func) for func in func_executables}
    pairs = list(permutations(funcs.keys(), 2))
    K = np.zeros((len(func_order), len(func_order)))
    human_readable_results = []

    # Ask gpt what pairs are subsumption pairs
    subsumption_pairs = await extract_subsumptions_from_source_code(funcs)
    for pair in subsumption_pairs:
        K[func_order[pair[0]], func_order[pair[1]]] = 1
        human_readable_results.append(
            {
                "func A": funcs[pair[0]],
                "func B": funcs[pair[1]],
                "A -> B": True,
                "asked_LLM": True,
            }
        )

    # "Fix" K with M and transitivity
    for pair in pairs:
        if pair in subsumption_pairs:
            continue

        run_pair = may_subsume(pair[0], pair[1], funcs, func_order, M)
        if run_pair == 0:
            K[func_order[pair[0]], func_order[pair[1]]] = 0
            human_readable_results.append(
                {
                    "func A": funcs[pair[0]],
                    "func B": funcs[pair[1]],
                    "A -> B": False,
                    "asked_LLM": False,
                }
            )
            continue
        elif run_pair == 2:
            K[func_order[pair[0]], func_order[pair[1]]] = 1
            human_readable_results.append(
                {
                    "func A": funcs[pair[0]],
                    "func B": funcs[pair[1]],
                    "A -> B": True,
                    "asked_LLM": False,
                }
            )
            continue

        # See if a subsumes b by transitivity
        # if there is some c such that a subsumes c and c subsumes b, then a subsumes b
        for c in func_order.keys():
            if (
                K[func_order[pair[0]], func_order[c]] == 1
                and K[func_order[c], func_order[pair[1]]] == 1
            ):
                print(f"Using transitivity based on {c}")
                K[func_order[pair[0]], func_order[pair[1]]] = 1
                human_readable_results.append(
                    {
                        "func A": funcs[pair[0]],
                        "func B": funcs[pair[1]],
                        "A -> B": True,
                        "asked_LLM": False,
                    }
                )
                break

    human_readable_results = pd.DataFrame(human_readable_results)

    return {"K": K, "human_readable_results": human_readable_results}
