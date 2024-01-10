import os
from constants import DEFAULT_DIRECTORY, DEFAULT_MODEL, DEFAULT_MAX_TOKENS, EXTENSION_TO_SKIP
from dotenv import load_dotenv
from glob import glob
from utils import get_file_paths, get_functions, get_file_content, get_function_name, get_until_no_space, num_tokens_from_string, truncate_text_tokens, len_safe_get_embedding, save_embedded_code
from codeagents import code_understanding_agent, code_error_detection_agent, code_testing_agent, code_optimization_agent, code_documentation_agent, code_algorithm_agent, code_design_agent, code_prompt_agent
from gptfunctions import ChatGPTAgent
import openai
from openai.embeddings_utils import get_embedding
import pandas as pd
import numpy as np
import json

from dotenv import load_dotenv
# Initialize OpenAI and GitHub API keys
openai.api_key = os.getenv('OPENAI_API_KEY')

tokenLimit = 2000

def chunk_and_summarize(code_file):
    chunks = 1
    code = get_file_content(code_file)
    if code is None:
        return None
    tokens = num_tokens_from_string(code)
    function_list = []
    docs = []
    if tokens < tokenLimit:
        doc_text = ChatGPTAgent.chat_with_gpt3(code, code_documentation_agent())
        docs.append({"doc": doc_text, "code": code, "filepath": code_file})  # dict
        print("tokens < limit. saving full code")
        docs.append({"doc": doc_text, "code": code, "filepath": code_file})  # dict
    else:
        funcs = list(get_functions(code))
        for func in funcs:
            potential_tokens = tokens + num_tokens_from_string(func)
            if potential_tokens < tokenLimit:
                function_list.append(func)
                tokens = potential_tokens
            else:
                print("Need to chunk the data but not lose track when doing multiple summaries") 
                function_list = [func]
                tokens = num_tokens_from_string(code)
        if function_list:
            doc = ChatGPTAgent.chat_with_gpt3(function_list, code_documentation_agent())
            docs.append(doc)
    return docs

def create_algorithms_and_design(all_docs):
    all_docs_string = json.dumps(all_docs)
    tokens = num_tokens_from_string(all_docs_string)
    algorithms = []
    designs = []
    docs_list = []
    if tokens < tokenLimit:
        algorithm = ChatGPTAgent.chat_with_gpt3(all_docs_string, code_algorithm_agent())
        algorithms.append(algorithm)
        design = ChatGPTAgent.chat_with_gpt3(all_docs_string, code_design_agent())
        designs.append(design)
    else:
        for doc in all_docs:
            doc_string = json.dumps(doc)
            potential_tokens = tokens + num_tokens_from_string(doc_string)
            if potential_tokens < tokenLimit:
                docs_list.append(doc_string)
                tokens = potential_tokens
            else:
                doc_list_string = json.dumps(docs_list)
                algorithm = ChatGPTAgent.chat_with_gpt3(doc_list_string, code_algorithm_agent())
                algorithms.append(algorithm)    
                design = ChatGPTAgent.chat_with_gpt3(doc_list_string, code_design_agent())
                designs.append(design)
                docs_list = [doc_string]
                tokens = num_tokens_from_string(all_docs_string)
        if docs_list:
            doc_list_string = json.dumps(docs_list)
            algorithm = ChatGPTAgent.chat_with_gpt3(doc_list_string, code_algorithm_agent())
            algorithms.append(algorithm)
            design = ChatGPTAgent.chat_with_gpt3(doc_list_string, code_design_agent())
            designs.append(design)
    return algorithms, designs

def create_prompts_from_algorithms_and_designs(algorithms, designs):
    prompts = []
    for algorithm, design in zip(algorithms, designs):
        prompt = "Algorithm: " + algorithm + "\nDesign: " + design
        prompts.append(prompt)
    return prompts


def main():
    load_dotenv()
    repo_url = os.getenv('REPO_URL')
    clone_dir = os.getenv('CLONE_DIR')
    file_paths = get_file_paths(clone_dir)
    code_files = [y for x in os.walk(clone_dir) for ext in ('*.py', '*.js', '*.cpp', '*.rs', '*.md', '*.txt') for y in glob(os.path.join(x[0], ext))]
    if len(code_files) == 0:
        print("Double check that you have downloaded the repo and set the code_dir variable correctly.")
    all_funcs = []
    all_docs = []
    for code_file in code_files:
        docs = list(chunk_and_summarize(code_file))
        funcs = list(get_functions(code_file))
        for func in funcs:
            all_funcs.append(func)
        for doc in docs:
            all_docs.append(doc)
    all_docs_string = json.dumps(all_docs)
    tokens = num_tokens_from_string(all_docs_string)
    if tokens < tokenLimit:
        print("tokens < limit with all docs. getting prompt")
        prompt = ChatGPTAgent.chat_with_gpt3(all_docs_string, code_prompt_agent())
        print(f"Prompt: " + prompt)
    else:
        algorithms, designs = create_algorithms_and_design(all_docs)  
        prompts = create_prompts_from_algorithms_and_designs(algorithms, designs)
        prompts_string = json.dumps(prompts)
        prompts_tokens = num_tokens_from_string(prompts_string)     
        if prompts_tokens < tokenLimit:
            prompt = ChatGPTAgent.chat_with_gpt3(prompts_string, code_prompt_agent())
            print(prompt)
        else:
            print("Need to chunk data for prompts") 
            print(prompts)
    print("Total number of functions:", len(all_funcs))
    save_embedded_code(all_funcs, clone_dir, "functions", "code")
    save_embedded_code(all_docs, clone_dir, "documentations", "doc")

    


if __name__ == "__main__":
    main()