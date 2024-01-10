import sys
print("Starting")
# Check if the openai module is installed.
try:
    import openai
except ImportError:
    print('openai module not found. Try running "pip3 install openai"')
    sys.exit(1)

import os
import argparse

# THIS IS PRETTY GOOD! 
# TODO Make it so that it takes in feedback from users from the comment section of github which then regenerates the read me with improvements
# Accept command line arguments for what project you want to generate a readme for. Leaving it as default generates the current directory

FILES_NOT_TO_INCLUDE = ['LICENSE', 'README.md', 'Pipfile', 'autoReadMe.py', 'myAutoReadMe.py']
STREAM = True

NUM_TOKENS = 1000

cur_dir_not_full_path = os.getcwd().split('/')[-1]
README_START =  f'# {cur_dir_not_full_path}\n## What is it?\n'

openai.api_key = openai.api_key = os.environ["OPENAI_API_KEY"]

def generate_completion(input_prompt, num_tokens):
    response = openai.Completion.create(engine='text-davinci-003', prompt=input_prompt, temperature=0.5, max_tokens=num_tokens, stream=STREAM, stop='===================\n')
    return response

def generate_completion_chatgpt(input_prompt, num_tokens):
    messages = [
            {'role': 'user', 
            'content': input_prompt}
            ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        max_tokens=num_tokens,
      messages=messages,
    )

    return response

# Read in the files in the current directory.
files_sorted_by_mod_date = sorted(os.listdir('.'), key=os.path.getmtime)
print(files_sorted_by_mod_date)


file_summaries = []
files_summarised = []
summarise_prompt = f"Summarise the following file with the following information - what is this file, what does it do and what other files are inputted to it. Write the summary in {NUM_TOKENS * 4} words or less. File: \n\n"
for file in files_sorted_by_mod_date:
    summarise_input = summarise_prompt
    if file not in FILES_NOT_TO_INCLUDE and not file.startswith('.') and not os.path.isdir(file) and not file.startswith('_') and not file.endswith('.json') and not file.endswith('.yaml') and not file.endswith('.lock'):
        summarise_input += "===================\n# " + file + ":\n"
        with open(file) as f:
            summarise_input += f.read() + "\n"
        summarised_file = generate_completion_chatgpt(summarise_input, 1000)["choices"][0]["message"]['content']
        summary = {
            'file': file,
            'summary': summarised_file # TODO: # generate summarisation
        } 
        file_summaries.append(summary)
        files_summarised.append(file)
        

for file in file_summaries:
    print(file['file'])
    print(file['summary'])
    print("\n\n\n\n\n")


other_files_in_directory = list(set(files_summarised) - set(files_sorted_by_mod_date))
# Create the initial prompt
prompt = f"Create a readMe file with the following sections: ```What is it, Requirements, How to run it, Examples.``` Write the ReadME in {NUM_TOKENS*4} words or less. \n\n"
prompt += "Assume files with the following filenames exist: \n " + str(other_files_in_directory) + "\n\n"
prompt += "Make the readMe based off the summarisation of the following files:\n\n "
gpt_input = prompt
# Now generate the readMe file
for file in file_summaries:
    gpt_input += "===================\n# " + file['file'] + ":\n"
    gpt_input += file['summary'] + "\n"

print("\n\n\n\n\n READ ME GENERATION \n\n\n")
#TODO generate summarisation from chatGPT
print(generate_completion_chatgpt(gpt_input, 1000)["choices"][0]["message"]['content'])
# print(gpt_input)
