import argparse
import numpy as np
import openai
import re 
import subprocess

import pygments
from pygments.lexers import PythonLexer
from pygments.formatters import Terminal256Formatter
pp = lambda s: pygments.highlight(s, PythonLexer(), Terminal256Formatter())

from env import OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY


# Helpers

def call_openai_api(prompt):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}], 
        temperature=0.5).choices[0].message["content"].strip()

def parse_code(code):
    code = code.strip()
    match = re.search(r"```(python)?\n(.*)```", code, re.DOTALL)
    if match:
        code = match.group(2).strip()
    return code

def run_code(code, ask=True):
    if (input("Run code? [y/n] ") if ask else "y") == "y":
        result = subprocess.run(['python', '-c', code], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return f"{result.stdout.decode()}\n{result.stderr.decode()}".strip()
    else:
        return None

def parse_tasks(tasks):
    task_list = []
    for task in tasks.split("\n"):
        tokens = task.strip().split(".", 1)
        if len(tokens) == 2:
            task = tokens[1].strip()
            if len(task) >= 3:
                task_list.append(task)
    return task_list

def str_tasks(tasks):
    return "\n".join([f"{idx+1}. {task}" for idx, task in enumerate(tasks)])


# LLM functions

def init_tasks(code, output):
    prompt = f"You are a rockstar Python programmer tasked to improve a piece of Python code written by a novice. Your goal is to make the code correct, short and efficient. Do not add boilerplate code.\n\nCurrent code:\n```python\n{code}```\n\n"
    if output:
        prompt += f"Current output:\n```{output}```\n\n"
    prompt += f"Provide an initial list of tasks to improve the code. Return the list as a bullet list, like:\n#. First task\n#. Second task."
    response = call_openai_api(prompt)
    return parse_tasks(response)

def execute_task(code, output, completed_tasks, task):
    prompt = f"You are a rockstar Python programmer who performs one task to improve a piece of Python code written by a novice. Your goal is to make the code correct, short and efficient. Do not add boilerplate code.\n\nCurrent code:\n```python\n{code}```\n\n"
    if output:
        prompt += f"Current output:\n```{output}```\n\n"
    prompt += f"Take into account these previously completed tasks:\n{str_tasks(completed_tasks)}\n\n"
    prompt += f"Your next task: {task}\n\nReturn the code WITHOUT making ANY other changes than those necessary for the current task. Keep tests if any. Return code ONLY. No verbose, no chat, no comments, no explanations."
    code = call_openai_api(prompt)
    return parse_code(code)

def create_tasks(code, output, completed_tasks, uncompleted_tasks, task):
    prompt = f"You are a rockstar Python programmer tasked to improve a piece of Python code written by a novice. Your goal is to make the code correct, short and efficient. Do not add boilerplate code.\n\nCurrent code:\n```python\n{code}```\n\n"
    if output:
        prompt += f"Current output:\n```{output}```\n\n"
    prompt += f"Take into account these previously completed tasks:\n{str_tasks(completed_tasks)}\n\n"
    prompt += f"Take into account these uncompleted tasks:\n{str_tasks(uncompleted_tasks)}\n\n"
    prompt += f"Return a short list of new tasks (0 to 3) that need to be completed to improve the code. Do not propose new tasks that overlap with the uncompleted tasks. Do not propose tasks that have already been completed. Return the list as a bullet list, like:\n#. First task\n#. Second task."
    response = call_openai_api(prompt)    
    return parse_tasks(response)

def prioritize_tasks(tasks, code, output, max_tasks=10):
    prompt = f"You are a task prioritization expert. You are tasked with cleaning the formatting of and reprioritizing the following list of tasks:\n{str_tasks(tasks)}\n\nPrioritize major bug fixes first. If needed, remove tasks that are redundant or no longer necessary. Keep {max_tasks} tasks at most. The ultimate goal is to make a piece of Python code correct, short and efficient.\n\nCurrent code:\n```python\n{code}```\n\n"
    if output:
        prompt += f"Current output:\n```{output}```\n\n"
    prompt += f"Return the list as a bullet list, like:\n#. First task\n#. Second task."
    response = call_openai_api(prompt)
    return parse_tasks(response)


# Thinking loop

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", help="Python file to fix and improve")
parser.add_argument("-a", "--autonomous", action="store_true", help="Autonomous mode", default=False)
parser.add_argument("-y", "--yes", action="store_true", help="Yes to all", default=False)
args = parser.parse_args()

code = parse_code(open(args.file, "r").read())
tasks = []
completed_tasks = []
i = 0

while len(tasks) > 0 or i == 0:
    # Print current state
    print(f"\n[{i}]", "="*80, "\n\n")
    print(f"CODE:\n{pp(code)}\n\n")

    output = run_code(code, ask=not args.yes)
    if output:
        print(f"OUTPUT:\n{output}\n\n")  

    if i > 0:
        # Create new tasks and reprioritize
        proposals = create_tasks(code, output, completed_tasks, tasks, task)
        tasks = prioritize_tasks(tasks + proposals, code, output)
    else:
        # Initialize tasks
        tasks = init_tasks(code, output)

    print(f"PENDING TASKS:\n{str_tasks(tasks)}\n\n")

    # Pop next task
    if args.autonomous:
        task = tasks.pop(0)
    else:
        task = tasks.pop(int(input("Task id? ").strip())-1)

    print(f"TASK:\n{task}\n\n")

    # Execute task
    code = execute_task(code, output, completed_tasks, task)
    completed_tasks.append(task)

    print(f"NEW CODE:\n{pp(code)}\n")

    # Continue?
    if not args.yes:
        if input("Continue? [y/n] ") == "n":
            break

    i += 1
