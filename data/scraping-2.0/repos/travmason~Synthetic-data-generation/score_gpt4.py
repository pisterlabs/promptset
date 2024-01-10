import os
import openai
from time import time,sleep
from dotenv import load_dotenv
import json
import re

try:
    load_dotenv()  # take environment variables from .env.
except Exception as oops:
    print("Issue with load_dotenv:" + oops)

import argparse

parser = argparse.ArgumentParser(description="Parse all the output of gpt_completions in a folder. Defaults to the last run.")
parser.add_argument('-i', '--input', type=str, help="Optional. Input folder. Defaults to the last run.")

args = parser.parse_args()

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()


def save_convo(text, topic):
    with open('finetuning/%s_%s.txt' % (topic, time()), 'w', encoding='utf-8') as outfile:
        outfile.write(text)

openai.api_key = os.getenv("OPENAI_API_KEY")
prompt_version = os.getenv("PROMPT_VERSION")
#base_prompt = open_file('score_prompt.txt')
base_prompt = open_file('score_prompt_counselling_competancys.txt')

def gpt4_completion(wdir, prompt, topic, engine='gpt-4', temp=1, top_p=1.0, tokens=3500, freq_pen=0.0, pres_pen=0, stop=['<<END>>']):
    max_retry = 5
    retry = 0
    while True:
        try:
            messages=[{"role": "user", "content": f'{prompt}'}]

            response = openai.ChatCompletion.create(
                model=engine,
                messages=messages,
                temperature=temp,
                max_tokens=tokens,
                top_p=top_p,
                frequency_penalty=freq_pen,
                presence_penalty=pres_pen,
                stop=stop)
            text = response['choices'][0]['message']['content'].strip()
            return text
        except Exception as oops:
            retry += 1
            if retry >= max_retry:
                return "GPT error: %s" % oops
            print(f'Error communicating with OpenAI: {oops}\n')
            sleep(0.25)

def quality_check(run):
    filelist = filter(lambda x: (x.endswith('.json')), os.listdir(f'gpt3_logs/{run}.run'))
    for file in filelist:
        current_topic = open(file, 'r')
        print(f'{current_topic}\n')

def score(filepath):
    prompt_arr = {
        'Prompt':[], 
        'Response': []
    }

    with open(filepath, 'r') as file:
        conversation = file.read()
    prompt = base_prompt.replace('<<CONVERSATION>>', conversation)
    prompt_arr['Prompt'].append(prompt)            
    # print(f'Prompt: {prompt}\n')
    response = gpt4_completion("", prompt, "score")
    prompt_arr['Response'].append(response)
    response = response.replace('\n', '')
    response = response.strip()

    return response

if __name__ == '__main__':
    directory = 'gpt3_logs'

    # Create a list of all the files in the current directory
    # we're going to use this to create a new directory for the current run
    # filelist = filter(lambda x: (x.endswith('.run')), os.listdir(directory))

    filelist = filter(lambda x: (x.endswith('.run')), os.listdir(directory))

    #check arguments to see if we have a specified directory to use
    if args.input:
        if not os.path.exists(os.path.join(directory, f"{args.input}.run")):
            print(f'Error: Directory {args.input}.run does not exist.')
            exit(1)
        highest_number = args.input
    else:
        # Find the highest numbered directory
        highest_number = 0
        for file in filelist:
            try:
                number = int(file.rstrip('.run'))
                if number > highest_number:
                    highest_number = number
            except ValueError:
                pass  # Ignore if the file name is not a number

    #set the new working directory based on the working directory name
    directory = os.path.join(directory, f"{highest_number}.run")

    # Directory path containing the JSON files
    directory_path = directory
    log_file = directory_path.split('/')[1].split('.')[0]
    log_file = log_file + '_score.log'
    print(f'Log file: {log_file}')
    print(f'Directory path: {directory_path}')
    print(f'Full path: {os.path.join(directory_path, log_file)}')

    response = []
    loopcount = 0
    text = ""

    # Iterate over each file in the directory
    for filename in os.listdir(directory_path):
        if filename.endswith('.txt'):
            # Construct the full file path
            file_path = os.path.join(directory_path, filename)
            print(f"Processing {file_path}")
            # Process the file
            text = score(file_path)                        
            print(f"Response: {text}")

            #text = text.split('{')[0].strip()
            #text = fixed_data = re.sub(r',(\s*})', r'\1', text)
            print(f"Response just before json.loads(): {text}")
            response.append(json.loads(text))

            #sleep(5)

        # if loopcount > 2:
        #     break
        # loopcount += 1

    with open(os.path.join(directory_path, log_file), 'w') as f:
        json.dump(response, f, ensure_ascii=False, indent=4)