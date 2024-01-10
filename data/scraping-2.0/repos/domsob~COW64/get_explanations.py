import random
import os
import openai
import open_ai_key
from tqdm import tqdm
from helper_functions import get_completion, remove_java_comments, mark_unchanged_lines, get_subfolder_names, create_directory_if_not_exists


openai.api_key = open_ai_key.api_key

prompt = """
In the given code, lines starting with an "o" indicate unchanged lines, lines starting with a "+" indicate added lines, and lines starting with a "-" indicate removed lines.
Please explain only the modifications made using the provided template:

Condition: Under what circumstances or conditions was the change necessary?
Consequence: What are the potential impacts or effects of this change?
Position: Where in the codebase was the change implemented?
Cause: What was the motivation for this change? Why was the previous implementation insufficient or problematic?
Change: How was the code or behavior being altered to address the identified condition or problem?

The code:

"""

model_name = 'gpt-3.5-turbo-16k'    # gpt-3.5-turbo-16k   # gpt-4
base_folder_generated = 'ARJAe'
base_folder_human = 'Human'
output_folder = 'responses'
temperature = 0.8
considered_problem_number = 30  
number_of_runs = 3
seed = 42

def write_prompt(base_folder, problem_name, prompt_prefix): 
    with open(base_folder + '/' + problem_name + '/long_diff.patch', "r") as file:
        contents = file.read()

    contents = remove_java_comments(contents.strip())
    contents = mark_unchanged_lines(contents.strip())

    temp_prompt = prompt_prefix + contents

    return temp_prompt

def make_calls_for_problem(problem_name, used_type, used_prompt, runs):
    for i in range(1, runs + 1):
        response = get_completion(used_prompt.strip(), model=model_name, temp=temperature)
        # response = output_folder + '/' + problem_name + '/' + used_type + '/' + 'run' + str(i) + '_' + used_type + '_' + problem_name + '.txt'

        output = '# REQUEST #####################################' + '\n\n' + used_prompt + '\n\n' + '# RESPONSE ####################################' + '\n\n' + response
    
        create_directory_if_not_exists(output_folder + '/' + problem_name + '/' + used_type)
        with open(output_folder + '/' + problem_name + '/' + used_type + '/' + 'run' + str(i) + '_' + used_type + '_' + problem_name + '.txt', "w") as file:
            file.write(output)

if __name__ == "__main__":

    if os.path.exists(output_folder):
        print('The results folder already exists. Please delete it before running this script.')
        exit()
    else:
        os.makedirs(output_folder)

    random.seed(seed)

    considered_problems = get_subfolder_names(base_folder_generated)
    random.shuffle(considered_problems)
    considered_problems = considered_problems[:considered_problem_number]

    for problem in tqdm(considered_problems, desc="Processed problems", unit="problem"):

        generated_prompt = write_prompt(base_folder_generated, problem, prompt)
        human_prompt = write_prompt(base_folder_human, problem, prompt)

        make_calls_for_problem(problem, 'generated', generated_prompt, number_of_runs)
        make_calls_for_problem(problem, 'human', human_prompt, number_of_runs)
