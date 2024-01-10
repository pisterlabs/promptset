import os, json
from datasets import load_dataset
from typing import Dict
from timeout_decorator import timeout
import gc
import re

from agent.self_doc_agent import SelfDocAgent
from openai import OpenAI

from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from rich import print

AGENT_MATCHER = {"SelfDocAgent": SelfDocAgent}


def extract_all_code_block_gpt(input_str: str) -> str:
    pattern = r"```python(.*?)```"
    matches = re.findall(pattern, input_str, re.DOTALL)

    return "\n".join([match.strip() for match in matches]) if matches else None


def delete_print_assert(code_text: str):
    lines = code_text.split("\n")
    new_lines = list()
    for i in lines:
        if i.strip().startswith("print("):
            continue
        # if i.strip().startswith("assert"):
        #    continue
        new_lines.append(i)

    new_code_text = "\n".join(new_lines)
    return new_code_text


def get_last_outermost_function_name(function_str):
    matches = re.findall(r"^def (\w+)", function_str, re.MULTILINE)
    if matches:
        return matches[-1]  # Return the last (outermost) function name
    return ""


@timeout(100, timeout_exception=TimeoutError)
def exec_with_timeout(import_str, full_test_code):
    env = {**locals()}
    code_to_exec = f"{import_str}\n{full_test_code}"
    try:
        exec(code_to_exec, env)
    except Exception as e:
        print(f"Error Type: {type(e).__name__}, Error Message: {e}")
        return False  # Return False if there's an error during execution, which mean incorrect
    return True  # Return True if executed without errors


import_str = """
import re
import math
from typing import List, Tuple, Optional

# this is for HumanEval38 (this should be preloaded for the test)
def encode_cyclic(s: str): 
    # split string to groups. Each of length 3.
    groups = [s[(3 * i):min((3 * i + 3), len(s))] for i in range((len(s) + 2) // 3)] 
    # cycle elements in each group. Unless group has fewer elements than 3. 
    groups = [(group[1:] + group[0]) if len(group) == 3 else group for group in groups] 
    return "".join(groups)
"""


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
        return data


def run_humaneval(
    agent: str = "SelfDocAgent",
    agent_config: Dict = {},
    TEST_FROM_HISTORY: bool = False,
):
    # human_eval
    human_eval_test = load_dataset("openai_humaneval")["test"]

    if TEST_FROM_HISTORY:
        memory_files = [
            os.path.join(f"./memory/{agent}/{agent_config['init']['model']}", i)
            for i in os.listdir(f"./memory/{agent}/{agent_config['init']['model']}")
            if "_True" in i or "_False" in i
        ]

        for _file in memory_files:
            problem_idx = int(_file.split("/")[-1].split(".json")[0].split("_")[1])
            test_code = human_eval_test[problem_idx]["test"]
            history = load_json(_file)
            assistant_answer = history["dialog"][-1]["content"]
            code_block = extract_all_code_block_gpt(assistant_answer)

            function_str = ""
            if (code_block is not None) and ("def" in code_block):
                function_str = code_block

            function_str = delete_print_assert(function_str)
            function_name = get_last_outermost_function_name(function_str)
            full_test_code = (
                f"{function_str}\n#-----------\n{test_code}\ncheck({function_name})"
            )

            is_correct = False
            try:
                is_correct = exec_with_timeout(import_str, full_test_code)
            except TimeoutError as e:
                timeout_flag = True
                print(f"Timeout with error msg : {e}")

            print(f"HumanEval/{problem_idx} {is_correct}")

        exit()

    # setup memory
    memory_bank = None

    correct, tested = 0, 0
    for problem in human_eval_test:
        task_id = problem["task_id"].replace("/", "_")
        problem_text = problem["prompt"].replace("    ", "\t")
        test_code = problem["test"]
        problem_text_refine = (
            f"Write a Python script to solve the following problem:\n{problem_text}\nEnsure the solution is verified by printing the expected output.",
        )[0]

        # Check already evaluated
        if os.path.exists(
            f"./memory/{agent}/{agent_config['init']['model']}/{task_id}_True.json"
        ):
            correct += 1
            tested += 1
            print(
                f"Problem {task_id} [green][correct][/green] **already evaluated** so skipped "
            )
            print(
                f"[bold blue]acc:[/bold blue] {correct}/{tested} = [bold green]{correct/tested:.2f}[/bold green]"
            )
            continue
        elif os.path.exists(
            f"./memory/{agent}/{agent_config['init']['model']}/{task_id}_False.json"
        ):
            if TEST_FROM_HISTORY:
                pass
            # correct += 1
            tested += 1
            print(
                f"Problem {task_id} [red][wrong][/red] **already evaluated** so skipped "
            )
            print(
                f"[bold blue]acc:[/bold blue] {correct}/{tested} = [bold green]{correct/tested:.2f}[/bold green]"
            )
            continue

        agent_instance = AGENT_MATCHER[agent](
            **agent_config["init"], memory=memory_bank
        )

        dialog = agent_instance.step(
            instruction=problem_text_refine,
            **agent_config["step"],
        )
        # resuse memory bank
        if memory_bank is None:
            memory_bank = agent_instance.get_memory()
            memory_bank.update_memory()

        agent_instance.close()
        gc.collect()

        print(
            f"agent done solving the task...\n[blue]Now extracting the infilling code lines[/blue]"
        )

        code_block = extract_all_code_block_gpt(dialog[-1]["content"])

        function_str = ""
        if (code_block is not None) and ("def" in code_block):
            function_str = code_block

        function_str = delete_print_assert(function_str)
        function_name = get_last_outermost_function_name(function_str)
        full_test_code = (
            f"{function_str}\n#-----------\n{test_code}\ncheck({function_name})"
        )

        # Print the full_test_code with syntax highlighting
        syntax = Syntax(
            # f"{programming_puzzle}\n{full_test_code}",
            f"{full_test_code}",
            "python",
            theme="monokai",
            line_numbers=True,
        )
        print(syntax)

        is_correct = False  # default is wrong
        timeout_flag = False

        try:
            is_correct = exec_with_timeout(import_str, full_test_code)
        except TimeoutError as e:
            timeout_flag = True
            print(f"Timeout with error msg : {e}")
        tested += 1

        if is_correct:
            correct += 1

        acc = (correct) / (tested)

        if is_correct:
            agent_instance.save_traj(traj_name=f"{task_id}_True.json")
        else:
            agent_instance.save_traj(traj_name=f"{task_id}_False.json")

        accuracy_text = Text(
            f"Accuracy: {correct}/{tested}[{tested}] = {acc:.2%} [{is_correct}]",
            style="bold blue",
        )
        panel = Panel(accuracy_text, title="Results", border_style="green")
        print(panel)


if __name__ == "__main__":
    agent_config = {
        "init": {
            "model": "gpt-4",
        },
        "step": {
            "USE_RETRIEVE": True,
            "USE_ENCODE": True,
            "ORGANIZE_MEMORY": True,
            "VERBOSE": True,
        },
    }

    run_humaneval(
        agent="SelfDocAgent",
        agent_config=agent_config,
        TEST_FROM_HISTORY=False,
    )
