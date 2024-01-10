"""LM Utilities."""

from typing import Any
import openai
import time
import tiktoken
import numpy as np

def get_openai_llm_response(prompt: str, config: dict[str, Any]) -> Any:
  openai.api_key = config['api_key']
  model_name = config['model_name']
  system_prompt = config['system_prompt']
  temperature = config['temperature']

  messages = [
      {'role': 'system', 'content': system_prompt},
      {'role': 'user', 'content': prompt},
  ]
  print(messages)
  responses = openai.ChatCompletion.create(
      model=model_name, messages=messages, temperature=temperature
  )
  return responses


def postprocess(responses, lm_config: dict[str, Any]) -> Any:
  # TODO: Implement more complicated postproc. Example: extracting only the blanks.
  # Extracting log probs
  if lm_config['api_type'] == 'openai':
    output = responses['choices'][0]['message']['content']
  else:
    raise NotImplementedError
  return {'answer': output}


def populate_prompt_fields(template: str, placeholder: str, excerpt: str):
  # This assumes that the template already contains any questions or examples
  # relevant to the prompt.
  # Can generalize this to a list of placeholder-excerpt lists eg: if examples
  # exist too.
  if placeholder not in template:
      print(f'Warning: {placeholder} not found')
  template = template.replace(placeholder, excerpt)
  return template

def get_number_of_tokens_in_line(line: str, model_name: str):
  encoding = tiktoken.encoding_for_model(model_name)
  return len(encoding.encode(line))

def llm_queries_with_retries(function, input_kwargs, error_type, num_retries=3):
  c = 0
  while c<num_retries:
    try:
      c+=1
      result = function(**input_kwargs)
    except error_type as e:
      print('Error: ', e)
      print(f"Retrying after {60*c} s")
      time.sleep(60*c)
    else:
      break
  return result # will throw result not defined error if you go through num_retries without getting any output

