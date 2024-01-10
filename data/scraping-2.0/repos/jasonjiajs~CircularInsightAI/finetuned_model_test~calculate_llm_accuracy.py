import openai
from openai import OpenAI
import pandas as pd
import numpy as np
import os
import json

client = OpenAI(api_key="API code")

data_path = "test_examples.jsonl"

# Get response from OpenAI API
def get_response(system_content, user_content, model):
    response = client.chat.completions.create(
        model=model,
        response_format={ "type": "json_object"},
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
            ],
        temperature=0,
        seed=0
    )
    return json.loads(response.choices[0].message.content)

# Load the dataset
with open(data_path, 'r', encoding='utf-8') as f:
    dataset = [json.loads(line) for line in f]

def compare_outputs(index):
    # Get message
    message = dataset[index]['messages']
    system_content = message[0]['content']
    user_content = message[1]['content']
    assistant_content = message[2]['content']

    assistant_content_default = get_response(system_content, user_content, model="gpt-3.5-turbo-1106")
    assistant_content_finetuned = get_response(system_content, user_content, model="ft:gpt-3.5-turbo-1106:personal::8e9YXb9p")

    print("--- Comparing outputs: ---")
    print("GPT response (ground truth): ", assistant_content)
    print("GPT response (default model): ", assistant_content_default)
    print("GPT response (finetuned model): ", assistant_content_finetuned)

    return assistant_content, assistant_content_default, assistant_content_finetuned

# assistant_content, assistant_content_default, assistant_content_finetuned = compare_outputs(1)

total_default_error = 0
total_tune_error = 0

for i in range(50):
  assistant_content, assistant_content_default, assistant_content_finetuned = compare_outputs(i)
  truth_dict = eval(assistant_content)
  default_dict = assistant_content_default
  tune_dict = assistant_content_finetuned

  default_error = abs(default_dict["relevance_problem"] - truth_dict["relevance_problem"])
  default_error += abs((default_dict["clarity_problem"] - truth_dict["clarity_problem"]))
  default_error += abs((default_dict["suitability_solution"] - truth_dict["suitability_solution"]))
  default_error += abs((default_dict["clarity_solution"] - truth_dict["clarity_solution"]))

  tune_error = abs(tune_dict["relevance_problem"] - truth_dict["relevance_problem"])
  tune_error += abs((tune_dict["clarity_problem"] - truth_dict["clarity_problem"]))
  tune_error += abs((tune_dict["suitability_solution"] - truth_dict["suitability_solution"]))
  tune_error += abs((tune_dict["clarity_solution"] - truth_dict["clarity_solution"]))

  total_default_error += default_error
  total_tune_error += tune_error

  print(total_default_error, total_tune_error, '/n')
