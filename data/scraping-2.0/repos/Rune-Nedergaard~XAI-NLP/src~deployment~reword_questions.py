import openai

from api_secrets import API_KEY

import openai
import os
import re
import pandas as pd
from pathlib import Path
import glob
from tqdm import tqdm



openai.api_key = API_KEY


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

failed_files = []

for filename in tqdm(os.listdir('data/translated_questions_correct'),total=len(os.listdir('data/translated_questions_correct'))):
    if not filename.endswith('.txt'):
        print(f"Skipping file {filename} (not a .txt file)")
        continue

    with open(os.path.join('data/translated_questions_correct', filename)) as f:
        example = f.read()

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
            if retry == 3:
                failed_files.append(filename)
                print(f"Request for file {filename} failed after 3 retries: {e}")
                break
            else:
                print(f"Request for file {filename} failed, retrying... ({retry}/3)")

if failed_files:
    print(f"The following files failed: {failed_files}")
