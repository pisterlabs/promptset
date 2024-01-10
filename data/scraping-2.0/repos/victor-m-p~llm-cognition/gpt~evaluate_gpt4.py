'''
VMP 2023-09-11:
This script is used to evaluate the data from GPT-4
'''
import openai 
import pandas as pd
import os 
from dotenv import load_dotenv
from tqdm import tqdm 
import re 
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

# setup 
outpath='../data/data_output/gpt4_eval/'
model='gpt-4'
temperature=0.8
frequency=0.0
presence=0.0
df = pd.read_csv('../data/data_cleaned/gpt4_shuffled.csv')

# function to create completion
@retry(wait=wait_random_exponential(min=1, max=200), stop=stop_after_attempt(10))
def create_completion(context, model, num_generations, 
                      max_tokens,
                      temperature, frequency, presence):
    completion = openai.ChatCompletion.create(
          model=model,
          messages=[{"role": "user", "content": context}],
          max_tokens=max_tokens,
          temperature=temperature,
          frequency_penalty=frequency,
          presence_penalty=presence,
          n=num_generations
          )
    return completion 

# setup
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# quick function to prepare prompt
def prepare_prompt(group):
    vignette = group['vignette'].iloc[0]
    id = group['id'].iloc[0]
    prompt = f'Here are six suggestions for what {id} could do. Please, rank them from best to worst. Your response should be a list of numbers, in the order from best to worst, where the best is first and the worst is last.\n'
    options = group['response_option'].tolist()
    options = [s.strip() for s in options]
    options = [f"{i+1}. {item}\n" for i, item in enumerate(options)]
    full_prompt = vignette + prompt + ''.join(options)
    condition = group['condition'].iloc[0]
    if condition == 'should': 
        full_prompt = re.sub('could', 'should', full_prompt)
    return full_prompt 

# sort the data and save prompts for reference
df = df.sort_values(by=['condition', 'id', 'iteration', 'shuffled'])
all_prompts = df.groupby(['condition', 'id', 'iteration']).apply(prepare_prompt)
prompts=all_prompts.to_frame(name='prompt').reset_index()
prompts.to_csv('../data/data_output/gpt4_eval/prompts.csv', index=False)

# loop over all generations, evaluate and save
data = []
for i, col in tqdm(prompts.iterrows()):
    condition = col['condition']
    id = col['id']
    iteration = col['iteration']
    prompt = col['prompt']
    # this really should (almost) never fail
    # gpt-4 should always (and to our knowledge does)
    # stick to the required format.
    # but just in case, we retry a few times
    max_attempts = 3  # set the maximum number of retries
    for attempt in range(max_attempts):
        try:
            completion = create_completion(prompt, 
                                           model, 
                                           1,
                                           20, # should always be 16 but to be safe
                                           temperature,
                                           frequency, 
                                           presence)
            completion = completion['choices'][0]['message']['content']
            shuffled = [int(x.strip()) for x in completion.split(',')]
            eval = range(1, 7)
            break  # Exit the loop if we successfully create 'shuffled'
        except ValueError:  # Handle the specific error that occurs when parsing fails
            if attempt == max_attempts - 1:  # No more attempts left
                print(f"Failed to generate completion for id {id}, iteration {iteration}.")
                continue  # Skip this row and move to the next
            else:
                print("Retrying...")
                
    for s, e in zip(shuffled, eval):
        row = {
            'condition': condition,
            'id': id,
            'iteration': iteration,
            'shuffled': s,
            'eval': e
        }
        data.append(row)

result_df = pd.DataFrame(data)
result_df.to_csv('../data/data_output/gpt4_eval/results.csv', index=False)