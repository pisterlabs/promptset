import os
import json
import re
import pandas as pd
from pathlib import Path
from openai import OpenAI
from tqdm import tqdm

# Load system prompt
with open('gpt_system_prompt.txt', 'r', encoding='utf-8') as file:
    gpt_system_prompt = file.read()

with open('system_prompt.txt', 'r', encoding='utf-8') as file:
    system_prompt = file.read()

print(gpt_system_prompt)

openai_client = OpenAI()

df = pd.read_json('eval_results_with_gpt_indented_fixed.json', orient='records')

gpt_responses = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    problem = row['problem']
    background = row['background']
    completion = openai_client.chat.completions.create(
      model="gpt-3.5-turbo-1106",
      temperature=0.8,
      top_p=0.95,
      messages=[
        {"role": "system", "content": gpt_system_prompt},
        {"role": "user", "content": f"{system_prompt}\n\n## Problem:\n{problem}\n\n## Background:\n{background}\n"}
      ]
    )
    gpt_responses.append(completion.choices[0].message.content)

df['gpt-3.5-turbo_solution'] = gpt_responses

df.to_json('eval_results_with_gpt.json', orient='records')
# sampling parameters
# sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2048)

# evaluation for model before fine-tuning
# before_finetune = LLM(model="mistralai/Mistral-7B-Instruct-v0.1", tensor_parallel_size=2)
# prompt_tokens = [prepare_mistral_prompt_tokens(problem, background) for problem, background in zip(df['problem'], df['background'])]
# before_finetune_outputs = before_finetune.generate(prompt_token_ids=prompt_tokens, sampling_params=sampling_params)
# before_finetune_solutions = [output.outputs[0].text for output in before_finetune_outputs]
# df['before_finetune_solution'] = before_finetune_solutions







