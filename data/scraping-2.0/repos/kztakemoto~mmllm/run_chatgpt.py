import openai
openai.api_key = "YOUR_API_KEY"

import pandas as pd
import random
from tqdm import tqdm
import time

from generate_moral_machine_scenarios import generate_moral_machine_scenarios

import argparse

#### Parameters #############
parser = argparse.ArgumentParser(description='Run ChatGPT')
parser.add_argument('--model', default='gpt-3.5-turbo-0613', type=str)
parser.add_argument('--nb_scenarios', default='3', type=int)
parser.add_argument('--random_seed', default='123', type=int)
args = parser.parse_args()

def chatgpt(system_cont, user_cont):
  try:
    res = openai.ChatCompletion.create(
      model=args.model,
      messages=[
            {"role": "system", "content": "Please respond to binary questions."},
            {"role": "system", "content": "{}".format(system_cont)},
            {"role": "user", "content": "{}".format(user_cont)}
        ],
    )

  except openai.error.OpenAIError as e:
    time.sleep(5)
    res = chatgpt(system_cont, user_cont)

  return res

random.seed(args.random_seed)
scenario_info_list = []
for i in tqdm(range(args.nb_scenarios)):
  # scenario dimension
  dimension = random.choice(["species", "social_value", "gender", "age", "fitness", "utilitarianism"])
  # Interventionism #########
  is_interventionism = random.choice([True, False])
  # Relationship to vehicle #########
  is_in_car = random.choice([True, False])
  # Concern for law #########
  is_law = random.choice([True, False])
  
  # generate a scenario
  system_content, user_content, scenario_info = generate_moral_machine_scenarios(dimension, is_in_car, is_interventionism, is_law)

  # obtain chatgpt response
  response = chatgpt(system_content, user_content)
  scenario_info['chatgpt_response'] = response['choices'][0]['message']['content']
  #print(scenario_info)

  scenario_info_list.append(scenario_info)

  if (i+1) % 100 == 0:
    df = pd.DataFrame(scenario_info_list)
    df.to_pickle('results_{}_scenarios_seed{}_{}.pickle'.format(args.nb_scenarios, args.random_seed, args.model))

df = pd.DataFrame(scenario_info_list)
df.to_pickle('results_{}_scenarios_seed{}_{}.pickle'.format(args.nb_scenarios, args.random_seed, args.model))