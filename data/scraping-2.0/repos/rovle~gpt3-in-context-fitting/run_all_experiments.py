"""
Just runs all the experiments in experiments_log.json which have
not yet been run.

Note: please uncomment the line which retrieves your OpenAI API key
if you wish to send requests to OpenAI API. This script costs actual
money if your experiments_log.json has any experiments which have not
been run, and you have an API key saved on your system.
"""


import json
import openai
import re
import os
from utils import textify_numbers
import argparse

with open('experiments_log.json', 'r') as file:
    experiments = json.loads(file.read())

#openai.api_key = os.getenv("OPENAI_API_KEY")

engines = ['ada', 'babbage', 'curie', 'davinci']
for engine in engines:
    for experiment_name in experiments.keys():
        if (f'output_test_raw_{engine}' in experiments[experiment_name].keys()
            or 'output_test_raw' in experiments[experiment_name].keys()):
            print(f"Have already performed {experiment_name} with this engine"
                    " skipping it now.")
            continue

        print(f"Running experiment {experiment_name} with engine {engine}. . .")
        
        experiment = experiments[experiment_name]
        experiment[f'response_{engine}'] = []
        experiment[f'output_test_raw_{engine}'] = []
        experiment[f'output_test_cleaned_{engine}'] = []
        
        for point in experiment['input_test']:
            point = textify_numbers(point)
            prompt_text = (
                experiment['input_text']
                + f'Input = {point}, output ='
            )
            response = openai.Completion.create(engine=engine,
                            prompt=prompt_text, max_tokens=6,
                            temperature=0, top_p=0)

            experiment[f'response_{engine}'].append(response)

            response_text = response['choices'][0]['text']
            experiment[f'output_test_raw_{engine}'].append(response_text)

            experiment[f'output_test_cleaned_{engine}'].append(
                int(
                    re.findall('-?\d+',response_text
                            )[0]
                    )
                )
                
        experiments[experiment_name] = experiment

with open('experiments_log.json', 'w') as file:
    json.dump(experiments, file, indent=4)
