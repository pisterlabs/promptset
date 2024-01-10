import openai
from api_secrets import API_KEY
import os
import re
import pandas as pd
from pathlib import Path
import glob
from tqdm import tqdm
import multiprocessing


openai.api_key = API_KEY
failed_files = []

# prompt_template = '''Transform questions into simple language that a typical person might use when searching for information online. The revised question should not be written as if it is addressed to the minister.

# Original question 1: Will the minister explain the governments foreign policy in the field of drugs in general and in relation to the negotiations in both the EU circle and in the Commission on Narcotic Drugs prior to the annual meeting in March 2014 and the preceding High Level Segment Session, including which areas and objectives the government prioritizes in the negotiations ?
# Revised question 1: What is the government's foreign policy on drugs?
# ##
# Original question 2: Is it the ministers assessment that the bureaucracy and the plethora of regulations imposed on Danish farmers have gone too far, when e.g. the guidance on fertilization and harmony rules requires 145 pages of review, and the minister considers that there are limits to how large the regulatory burdens and application requirements for e.g. obtaining EU funding, it is reasonable to impose on ordinary people practicing their liberal professions?
# Revised question 2: What is the effect of regulations and requirements on Danish farmers?
# ##
# Original question 3: [INSERT EXAMPLE HERE]
# Revised question 3:'''

#revised zero-shot prompt
prompt_template = '''Transform the question into simple language that a typical person might use when searching for information online. The revised question should not be written as if it is addressed to the minister.

Original question: [INSERT EXAMPLE HERE]
Revised question:'''

skipped_files = 0
processed_files = 0

def process_file(filename):
    if not filename.endswith('.txt'):
        print(f"Skipping file {filename} (not a .txt file)")
        return
    
    if os.path.exists(os.path.join('data/questions_rephrased', filename)):
        print(f"Skipping file {filename} (file already exists in questions_rephrased folder)")
        return
    try:
        with open(os.path.join('data/translated_questions_correct', filename)) as f:
            example = f.read()
    except UnicodeDecodeError:
        print(f"Skipping file {filename} (UnicodeDecodeError)")
        return

    retry = 0
    while retry < 10:
        prompt = prompt_template.replace("[INSERT EXAMPLE HERE]", example)
        try:
            response = openai.Completion.create(engine="text-davinci-003", prompt=prompt, max_tokens=60, temperature=0, stop=["?", ".", "\n"])
            response_text = response["choices"][0]["text"]
            response_text = response_text[1:] + "?"
            with open(os.path.join('data/questions_rephrased', filename), 'w') as f:
                f.write(response_text)
            break
        except Exception as e:
            retry += 1
            if retry == 10:
                print(f"Request for file {filename} failed after 10 retries: {e}")
                break
            else:
                print(f"Request for file {filename} failed, retrying... ({retry}/10)")


if __name__ == '__main__':
    num_processes = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_processes)
    filenames = os.listdir('data/translated_questions_correct')
    args = [filename for filename in filenames]
    results = pool.map(process_file, args)
    pool.close()
    pool.join()

