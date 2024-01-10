import openai
import time
from tqdm import tqdm
import random
import requests
from requests.exceptions import ConnectionError

openai.api_key = 'API_KEY'

def create_prompt(prompt_prefix, query, options_list):
  query = "Query: " + query + " \n\n"
  options = "Options: \n"

  for i in range(len(options_list)):
    options += str(i) + ". " + options_list[i] + "\n"

  prompt_suffix = "\nOption: "

  full_prompt = prompt_prefix + query + options + prompt_suffix
  return full_prompt

def create_fs_prefix(prompt_prefix, train_data, prompt_size):
  # generates sample text
  sample_data = random.sample(train_data, prompt_size)
  prompt_text = prompt_prefix

  for s in sample_data:
    query = "Query: " + s["query"] + " \n\n"
    options = "Options: \n"
    options_list = [val for val in s['options'].values()]
    # shuffle order of options in prompt
    random.shuffle(options_list)

    for i in range(len(options_list)):
      options += str(i) + ". " + options_list[i] + "\n"

    prompt_suffix = "\nOption: " + s['options'][s['answer']] + "\n\n"
    full_prompt = query+options+prompt_suffix
    prompt_text += full_prompt
  
  return prompt_text

# gets response from API
def get_response(prompt):
  tries = 5
  while True:
    tries -= 1
    try:
      response = openai.Completion.create(
                  model="text-davinci-003",
                  prompt=prompt,
                  temperature=0,
                  max_tokens=60
                )
      break
    except ConnectionError as err:
      if tries == 0:
        raise err
      else:
        time.sleep(5)

  #print(response)
  return response.choices[0].text

def gpt3_pred(train_data, test_data, prompt_size= 5, fewshot=False):
  # prompt instruction
  prefix = "Given the recipe query and five options, choose the option that best satisfies the query: \n\n"
  if fewshot:
    prefix = create_fs_prefix(prefix, train_data, prompt_size)
  
  predictions = []
  N = len(test_data)
  for i in tqdm(range(0, N, 1)):
    # limit API requests per minute
    if (i + 1) % 10 == 0:
      time.sleep(60)

    sample = test_data[i]
    options_list = [val for val in sample['options'].values()]
    query = sample['query']
    correct_answer = sample['options'][sample['answer']]

    prompt = create_prompt(prefix, query, options_list)
    # get API response
    answer = get_response(prompt)

    # if generated text response has correct answer description
    if correct_answer in answer:
      predictions.append(correct_answer)
    else:
      predictions.append(answer)
  
  return predictions