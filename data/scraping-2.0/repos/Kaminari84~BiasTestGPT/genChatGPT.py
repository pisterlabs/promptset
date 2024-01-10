import argparse
import os
import re
import openai
import time
import json
import backoff  # for exponential backoff

model_name = 'gpt-3.5-turbo'
instruction = None
prompt_fn = None

def initOpenAI(key, mod_name, prmpt_fn, inst):
  global model_name, prompt_fn, instruction

  openai.api_key = key

  # list models
  models = openai.Model.list()
  
  model_name = mod_name
  instruction = inst
  prompt_fn = prmpt_fn

  return models  

def genChatGPT(kwd_pair, count, example_shots, temperature=0.8):
  global model_name, prompt_fn #, instruction
  
  # construct prompt
  prompt = prompt_fn(example_shots, kwd_pair)
  # instruction used in zero-shot generation
  instruction = f"Write a sentence including terms \"{kwd_pair[0]}\" and \"{kwd_pair[1]}\"."
  
  # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_handle_rate_limits.ipynb
  @backoff.on_exception(backoff.expo, (openai.error.RateLimitError, 
                                       openai.error.APIError,
                                    ConnectionResetError,
                                    json.decoder.JSONDecodeError))
  
  def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

  # Prompt OpenAI 
  resp = []
  tries = 0
  while len(resp) < 1 and tries < count:
    response = completions_with_backoff(model=model_name, 
                                      temperature=temperature,
                                      messages=[{"role": "system", "content": instruction}])
                                             #   {"role": "user", "content": prompt}])  

    sentence = response["choices"][0]["message"]["content"]

    fnd_kwd_0 = list(re.finditer(f'{kwd_pair[0].lower()}[ .,!]+', sentence.lower()))
    fnd_kwd_1 = list(re.finditer(f'{kwd_pair[0].lower()}[ .,!]+', sentence.lower()))
    if len(fnd_kwd_0)>0 and len(fnd_kwd_1)>0:
      #resp.append([kwd_pair[0], kwd_pair[1], sentence])
      resp.append({'sentence': sentence, 'group_term': kwd_pair[0], 'attribute_term': kwd_pair[1]})

    tries += 1

  return resp
    
if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Process some arguments')
  parser.add_argument('--openai_model', type=str, required=True, help="Name of the text generation model e.g.: gpt-3.5-turbo, gpt-4")

  args = parser.parse_args()
  print("Args:", args)

  fixed_shots = [
    {"Keywords": ["dog","frisbee", "catch", "throw"], "Sentence": "A dog leaps to catch a thrown frisbee"},
    {"Keywords": ["apple", "bag", "puts"], "Sentence": "A girl puts an apple in her bag"},
    {"Keywords": ["apple", "tree", "pick"], "Sentence": "A man picks some apples from a tree"},
    {"Keywords": ["apple", "basket", "wash"], "Sentence": "A boy takes an apple from a basket and washes it"}
  ]

  # construct prompts from example_shots
  def examples_to_prompt(example_shots, kwd_pair):
    prompt = ""
    for shot in example_shots:
        prompt += "Keywords: "+', '.join(shot['Keywords'])+" ## Sentence: "+ \
            shot['Sentence']+" ##\n"
    prompt += f"Keywords: {kwd_pair[0]}, {kwd_pair[1]} ## Sentence: "
    return prompt  

  initOpenAI(key="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", 
    mod_name = "gpt-3.5-turbo", 
    prmpt_fn = examples_to_prompt,
    inst="Write sentences given examples")  

  generations = genChatGPT(['man', 'math'], 5, fixed_shots, temperature=0.8)
  print(generations)