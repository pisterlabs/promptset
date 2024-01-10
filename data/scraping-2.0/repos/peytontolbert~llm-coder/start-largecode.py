# Import necessary modules
import openai
import os
import sys
import time
import json
import re
import ast
from constants import DEFAULT_DIRECTORY, DEFAULT_MODEL, DEFAULT_MAX_TOKENS
from utils import clean_dir, write_file, get_file_content, get_file_paths, get_functions, chunk_and_summarize, num_tokens_from_string
from codingagents import clarifying_agent, algorithm_agent, coding_agent, debug_agent, file_code_agent, unit_test_agent
from glob import glob
from openai.embeddings_utils import get_embedding
import pathlib
import pandas as pd
from db import DB, DBs
import numpy as np
import traceback
from dotenv import load_dotenv
# Initialize OpenAI and GitHub API keys
openai.api_key = os.getenv('OPENAI_API_KEY')

tokenLimit = 8000
# Initialize a session with OpenAI's chat models
def chat_with_gpt3(systemprompt, prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": systemprompt},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9
    )
    return response['choices'][0]['message']['content']

def save_to_local_directory(repo_name, functions):
    # Check if the directory already exists
    if not os.path.exists(repo_name):
        # If not, create it
        os.makedirs(repo_name)
    
    # Create a new file in the directory to hold all the functions
    file_path = os.path.join(repo_name, "functions.py")
    with open(file_path, "w") as f:
        # Write all the functions to the file
        for function in functions:
            f.write(function)
            f.write("\n\n")

def clarify(prompt):
    while True:
        new_prompt = prompt
        clarifying_prompt = clarifying_agent()
        clarifying_prompt += (
            '\n\n'
            'Is anything unclear? If yes, only answer in the form:\n'
            '{remainingunclear areas} remaining questions. \n'
            '{Next question}\n'
            'If everything is sufficiently clear, only answer "no".'
        )
        clarifying_questions = chat_with_gpt3(clarifying_prompt, prompt)
        print(clarifying_questions)
        user_input = input('(answer in text, or "q" to move on)\n')
        new_prompt += user_input
        print()

        if not user_input or user_input.strip().lower() == 'q':
            break
    return new_prompt


def filter_filepaths(filepaths):
    filepaths_list = ast.literal_eval(filepaths)
    return [fp.lstrip('/') for fp in filepaths_list]

def generate_filepaths(prompt):
    systemprompt = f"""You are an AI developer who is trying to write a program that will generate code for the user based on their intent.
    When given their intent, create a complete, exhaustive list of filepaths that the user would write to make the program.
    Only list the filepaths you would write, and return them as a python array of strings. 
    do not add any other explanation, only return a python array of strings."""
    result = chat_with_gpt3(systemprompt, prompt)
    print(result)
    return result

def generate_filecode(clarifying_results, filepaths_string, shared_dependencies=None, prompt=None):
    print("generating code")
    prompt = f"""
    We have broken up the program into per-file generation. 
    Now your job is to generate only the code for the file {filepaths_string}. 
    Make sure to have consistent filenames if you reference other files we are also generating.
    
    Remember that you must obey 3 things: 
       - you are generating code for the file {filepaths_string}
       - do not stray from the names of the files and the shared dependencies we have decided on
       - follow the {clarifying_results} laid out in the previous steps.
    
    Bad response:
    ```javascript 
    console.log("hello world")
    ```
    
    Good response:
    console.log("hello world")
    
    Begin generating the code now.

    """
    systemprompt = file_code_agent(filepaths_string, shared_dependencies)
    filecode = chat_with_gpt3(systemprompt, prompt)
    print(filecode)
    return filecode


def generate_shared_dependencies(prompt, filepaths_string):
    systemprompt = f"""You are an AI developer who is trying to write a program that will generate code for the user based on their intent.
                
            In response to the user's prompt:

            ---
            the app is: {prompt}
            ---
            
            the files we have decided to generate are: {filepaths_string}

            Now that we have a list of files, we need to understand what dependencies they share.
            Please name and briefly describe what is shared between the files we are generating, including exported variables, data schemas, id names of every DOM elements that javascript functions will use, message names, and function names.
            Exclusively focus on the names of the shared dependencies, and do not add any other explanation.
    """
    result = chat_with_gpt3(systemprompt, prompt)
    print(result)
    return result

def debug_code(directory):
    extensions = ['py', 'html', 'js', 'css', 'c', 'rs']
    while True:
        code_files = []

        for extension in extensions:
            code_files.extend(y for x in os.walk(directory) for y in glob(os.path.join(x[0], f'*.{extension}')))
        print("Total number of py files:", len(code_files))
        if len(code_files) == 0:
            print("Double check that you have downloaded the repo and set the code_dir variable correctly.")

        all_funcs = []
        unit_tests = []
        for code_file in code_files:
            funcs = list(get_functions(code_file))
            code_tokens_string = json.dumps(code_file)
            code_tokens = num_tokens_from_string(code_tokens_string)
            if code_tokens < tokenLimit:
                unit_test = unit_test_agent(code_file)
            else:
                for func in funcs:

                    unit_test_prompt = unit_test_agent()
                    unit_test = chat_with_gpt3(unit_test_prompt, func)
                    unit_tests.append(unit_test)
            for func in funcs:
                all_funcs.append(func)
        all_funcs_string = json.dumps(all_funcs)
        print("Total number of functions:", len(all_funcs))
        df = pd.DataFrame(all_funcs)
        df['code_embedding'] = df['code'].apply(lambda x: get_embedding(x, engine="text-embedding-ada-002")) 
        df['filepath'] = df['filepath'].apply(lambda x: x.replace(directory, ""))
        df.to_csv("functions.csv", index=True)
        df.head()
        debug_code_agent = chat_with_gpt3(debug_agent, all_funcs_string)

        if not debug_code_agent or debug_code_agent.strip().lower() == 'no':
            break
        else:
            print(debug_code_agent)


# Main function
def main(prompt, directory=DEFAULT_DIRECTORY, model=DEFAULT_MODEL, file=None):
    if prompt.endswith(".md"):
        with open(prompt, "r") as f:
            prompt = f.read()
    print("Hello, I am your local AI developer! You said you wanted:")
    print("\033[92m" + prompt + "\033[0m")
    # Get the repo name from the user
    repo_name = input("Enter the name for the new directory: ")
    directory = os.path.join(directory, repo_name)
    prompt_string = json.dumps(prompt)
    new_prompt = clarify(prompt_string)
    new_prompt_string = json.dumps(new_prompt)
    print(prompt_string)
    filepaths = generate_filepaths(new_prompt_string)
    print(filepaths)
    list_actual = []
    try:
        list_actual = ast.literal_eval(filepaths)
        shared_dependencies = None
        if os.path.exists("shared_dependencies.md"):
            with open("shared_dependencies.md", "r") as f:
                shared_dependencies = f.read()
        
        if file is not None:
            print("File", file)
            filecode = generate_filecode(file, new_prompt_string, filepaths, shared_dependencies)
            write_file(filepaths, filecode, directory)
        else:
            clean_dir(directory)

            shared_dependencies = generate_shared_dependencies(prompt, filepaths)
            write_file("shared_dependencies.md", shared_dependencies, directory)

            for filepaths in list_actual:
                filecode = generate_filecode(new_prompt_string, filepaths, shared_dependencies)
                write_file(filepaths, filecode, directory)
        debug_code(directory)
    except Exception as e:
        print("Failed to parse result")
        print(f"Type: {type(e).__name__}")
        print(f"Message: {str(e)}")
        traceback.print_exc()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        if not os.path.exists("prompt.md"):
            print("Please provide a prompt file or a prompt string")
            sys.exit(1)
        else:
            prompt = "prompt.md"

    else:
        prompt = sys.argv[1]

    directory = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DIRECTORY
    file = sys.argv[3] if len(sys.argv) > 3  else None
    main(prompt, directory, file)