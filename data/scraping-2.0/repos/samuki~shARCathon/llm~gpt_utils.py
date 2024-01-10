import llm.secret as secret
import config

import openai
import os
import json
import shutil
import re
from scipy.sparse import csr_matrix


openai.api_key = secret.API_KEY

COLOR_NUMBER_DICT = {
    "0": "black",
    "1": "white",
    "2": "green",
    "3": "brown",
    "4": "yellow",
    "5": "blue",
    "6": "purple",
    "7": "pink",
    "8": "red",
    "9": "orange"
}

NUMBER_WORD_DICT = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine"
}

NUMBER_BINARY_DICT = {
    "0": "0",
    "1": "1",
    "2": "10",
    "3": "11",
    "4": "100",
    "5": "101",
    "6": "110",
    "7": "111",
    "8": "1000",
    "9": "1001"
}

NUMBER_CHAR_DICT = {
    "0": "a",
    "1": "b",
    "2": "c",
    "3": "d",
    "4": "e",
    "5": "f",
    "6": "g",
    "7": "h",
    "8": "i",
    "9": "j"
}

NUMBER_LEET_DICT = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "E",
    "4": "h",
    "5": "S",
    "6": "b",
    "7": "T",
    "8": "B",
    "9": "g"
}

NUMBER_SP_CHAR_DICT = {
    "0": "!",
    "1": "@",
    "2": "#",
    "3": "$",
    "4": "%",
    "5": "^",
    "6": "&",
    "7": "*",
    "8": "=",
    "9": "+"
}

def extract_result_text(result):
    if config.OPENAI_ENDPOINT == "Chat":
        return result["choices"][0]["message"]["content"]
    else:
        return result["choices"][0]["text"]

def prompt_gpt_completion(msg, model=config.GPT_MODEL):
    # Use OpenAI API
    # ChatCompletion
    completion = openai.Completion.create(
        model = model,
        temperature = config.TEMPERATURE,
        max_tokens = config.MAX_TOKENS,
        logprobs = config.LOG_PROBS,
        top_p = config.TOP_P,
        prompt = msg
    )
    return completion


def prompt_gpt_chat(user, system=False, model=config.GPT_MODEL):
    # Use OpenAI API
    # Split system and user for for API call
    if system:
        message = [{"role": "user", "content": user}, {"role": "system", "content": system}]
    else:
        message = [{"role": "user", "content": user}]
    completion = openai.ChatCompletion.create(
        model = model,
        temperature = config.TEMPERATURE,
        max_tokens = config.MAX_TOKENS,
        top_p = config.TOP_P,
        messages = message
    )
    return completion

def prompt_gpt(msg, system=False, model=config.GPT_MODEL):
    if config.OPENAI_ENDPOINT == "Chat":
        completion = prompt_gpt_chat(msg, system=system, model=model)
    elif config.OPENAI_ENDPOINT == "Completion":    
        completion = prompt_gpt_completion(msg, model=model)
    else:
        print("Endpoint not found")
    return completion

def preprocess_representation(prompt):
    if config.REPLACE_COMMA:
        prompt = prompt.replace(',', '')
    if config.REPLACE_SPACE:
        while re.search(r'(\d) (\d)', prompt):
            prompt = re.sub(r'(\d) (\d)', r'\1\2', prompt)
    if config.REPLACE_SPACE2:
        prompt = prompt.replace(', ', ',')
    if config.SEMICOLON:
        prompt = prompt.replace('] [', '; ').replace("[[", ";;").replace("]]", ";;")
    if config.BRACKETS:
        prompt = prompt.replace("[", "'").replace("]", "'").replace("[[", "''").replace("]]", "''")
    if config.REPLACE_NUMBER_COLOR:
        for key, value in COLOR_NUMBER_DICT.items():
            prompt = prompt.replace(key, value)
    if config.REPLACE_NUMBER_WORD:
        for key, value in NUMBER_WORD_DICT.items():
            prompt = prompt.replace(key, value)
    if config.REPLACE_NUMBER_BINARY:
        for key, value in NUMBER_BINARY_DICT.items():
            prompt = prompt.replace(key, value)
    if config.REPLACE_NUMBER_CHAR:
        for key, value in NUMBER_CHAR_DICT.items():
            prompt = prompt.replace(key, value)
    if config.REPLACE_NUMBER_LEET:
        for key, value in NUMBER_LEET_DICT.items():
            prompt = prompt.replace(key, value)
    if config.REPLACE_NUMBER_SP_CHAR:
        for key, value in NUMBER_SP_CHAR_DICT.items():
            prompt = prompt.replace(key, value)
    return prompt

def preprocess_training(task):
    intro = "Here is a training example.\n"
    train_string = "Examples: "
    for example in task['train']:
        train_string += f"input: {str(example['input'])} output: {str(example['output'])} \n"
    divider = ""
    test_string = f"Test: input: {str(task['test']['input'])} output: {str(task['test']['output'])} \n"
    return preprocess_representation(intro+ train_string + divider + test_string)


def preprocess_prompt_step_by_step(task):
    intro = "Do the following:\nWhat is the step by step description of the input/output relation that holds for all example input/output pairs?\n"
    train_string = "Examples: "
    for example in task['train']:
        train_string += f"input: {str(example['input'])} output: {str(example['output'])} \n"
    divider = "You now have all the information to solve the task. Apply this description to the test input and write you answer as 'output: '\n"
    test_string = f"Test: input: {str(task['test']['input'])} output:"
    return preprocess_representation(intro+ train_string + divider + test_string)

def preprocess_prompt(task):
    intro = "Continue the pattern"
    train_string = "Examples: "
    for example in task['train']:
        train_string += f"input: {str(example['input'])} output: {str(example['output'])} \n"
    divider = ""
    test_string = f"Test: input: {str(task['test']['input'])} output:"
    return preprocess_representation(intro+ train_string + divider + test_string)


def preprocess_self_correction(task):
    intro = "Do the following:\nWhat is the step by step description of the input/output relation that holds for all example input/output pairs?\n"
    train_string = "Examples: "
    for example in task['train']:
        train_string += f"input: {str(example['input'])} output: {str(example['output'])} \n"
    #divider = "You will now receive the test input and a possible test output. Combine both outputs to find errors and create the correct final output.\n"
    divider = ""
    test_string = f"Test: input: {str(task['test']['input'])} output:"
    return preprocess_representation(intro+ train_string + divider + test_string)


def preprocess_self_correction_two(task):
    intro = "Do the following:\nWhat is the step by step description of the input/output relation that holds for all example input/output pairs?\n"
    train_string = "Examples: "
    for example in task['train']:
        train_string += f"input: {str(example['input'])} output: {str(example['output'])} \n"
    divider = "You will now receive the test input and two possible test outputs. Combine both outputs to find errors and create the correct final output.\n"
    test_string = f"Test: input: {str(task['test']['input'])} output 1:"
    return preprocess_representation(intro+ train_string + divider + test_string)


def preprocess_prompt_create_description(task):
    intro = "Do the following:\nWhat is the step by step pattern that holds for all example input/output pairs?\n"
    train_string = "Examples: "
    for example in task['train']:
        train_string += f"input: {str(example['input'])} output: {str(example['output'])} \n"
    divider = "Create a step-by-step description of the pattern and how to apply it to the test input.'\n"
    test_string = f"Test: input: {str(task['test']['input'])} output:"
    return preprocess_representation(intro+ train_string + divider + test_string)


def preprocess_prompt_use_description(task, task_name):
    print("task ", task)
    intro = "Do the following:\nWhat is the step by step pattern that holds for all example input/output pairs?\n"
    train_string = "Examples: "
    for example in task['train']:
        train_string += f"input: {str(example['input'])} output: {str(example['output'])} \n"
    divider = "Here is a step-by-step description for the task:'\n"
    with open(config.DESCRIPTION_PATH.resolve() / f'{task_name}_out.json', 'r') as file:
        description = extract_result_text(json.load(file)['output'])
    divider_2 = "\nApply this description to the test input and write you answer as 'output: '\n'"       
    test_string = f"Test: input: {str(task['test']['input'])} output:"
    return preprocess_representation(intro+ train_string + divider + description+ divider_2+ test_string)
   
   
def naive_postprocessing(prediction):
    """Performs simple post-processing on model prediction"""
    return prediction.split(':')[-1].replace("\n", "").strip()


def compress(matrix):
    shape = (len(matrix), len(matrix[0])) if matrix else (0, 0)
    flat_list = [item for sublist in matrix for item in sublist]
    compressed = [shape]
    count = 1
    for i in range(1, len(flat_list)):
        if flat_list[i] == flat_list[i-1]:
            count += 1
        else:
            compressed.append((flat_list[i-1], count))
            count = 1
    compressed.append((flat_list[-1], count))
    return compressed


def decompress(compressed):
    shape = compressed[0]
    compressed = compressed[1:]
    decompressed = []
    for value, count in compressed:
        decompressed.extend([value]*count)
    return [decompressed[i:i + shape[1]] for i in range(0, len(decompressed), shape[1])]


def compress_matrix(json_task):
    train = []
    for example in json_task['train']:
        train.append({'input': compress(example['input']), 'output': compress(example['output'])})
    json_task['train'] = train
    json_task['test'] = {'input': compress(json_task['test']['input']), 'output': compress(json_task['test']['output'])}
    return json_task
    

def sparsify(json_task):
    train = []
    for example in json_task['train']:
        input_m = csr_matrix(example['input']).tocoo()
        output_m = csr_matrix(example['output']).tocoo()
        train.append({'input': [input_m.row.tolist(), input_m.col.tolist(), input_m.data.tolist()], 'output': [output_m.row.tolist(), output_m.col.tolist(), output_m.data.tolist()]})
    json_task['train'] = train
    test_input_m = csr_matrix(json_task['test']['input']).tocoo()
    test_output_m = csr_matrix(json_task['test']['output']).tocoo()
    json_task['test'] = {'input': [test_input_m.row.tolist(), test_input_m.col.tolist(), test_input_m.data.tolist()], 'output': [test_output_m.row.tolist(), test_output_m.col.tolist(), test_output_m.data.tolist()]}
    return json_task
    
 
def get_task(json_task, task_name, self_correction=False, training=False, describe=False):
    # ensure only one test output  
    json_task['test'] = json_task['test'][0]
    if config.SPARSE_MATRIX:
        json_task = sparsify(json_task)
    if config.COMPRESS:
        json_task = compress_matrix(json_task)
    if training:
        return preprocess_training(json_task)
    json_task['test']['output'] = ''
    if config.CREATE_DESCRIPTION:
        return preprocess_prompt_create_description(json_task)
    if config.USE_DESCRIPTION:
        return preprocess_prompt_use_description(json_task, task_name)
    if self_correction:
        return preprocess_self_correction(json_task)
    else:
        return preprocess_prompt(json_task)


def get_prompt(json_task):
    preamble = config.PROMPT_TEMPLATE
    return preamble + '\n\n' + str(get_task(json_task))


def get_directory():
    dataset = str(config.PATH_SELECTION.resolve()).split('/')[-1]
    replace_comma = 'replace_comma' if config.REPLACE_COMMA else 'no_replace_comma'
    return f"results/{dataset}/{replace_comma}/{config.GPT_MODEL}"


def save_gpt_results(task_name, prompt, result):
    directory = get_directory()
    # check if directory exists and create otherwise
    if not os.path.exists(directory):
        os.makedirs(directory)
    # copy config.py to the new directory
    shutil.copyfile('config.py', os.path.join(directory, 'config.py'))
    # create a json output file
    output_file_name = os.path.join(directory, task_name + "_out.json")
    # data to be written
    data = {"prompt": prompt, "output": result}
    # writing to json file
    with open(output_file_name, 'w') as outfile:
        json.dump(data, outfile)



