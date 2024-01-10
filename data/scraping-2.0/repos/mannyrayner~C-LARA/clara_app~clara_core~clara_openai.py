"""
This module contains utility functions for OpenAI functionality.
The functions assume that a valid license key is in the environment variable OPENAI_API_KEY

1. print_openai_models(). Print a list of all OpenAI models available for this license key.
2. cost_of_gpt4_api_call(messages, response). Return the cost in dollars of a gpt-4 API call.
"""

from . import clara_utils

import openai
import tiktoken
import os

config = clara_utils.get_config()

openai.api_key = os.getenv("OPENAI_API_KEY")

def print_openai_models():
    """List all available models"""
    models = openai.Model.list()

    for model in models['data']:
        print(f"Model ID: {model.id}")

def cost_of_gpt4_api_call(messages, response_string, gpt_model='gpt-4'):
    """Returns the cost in dollars of an OpenAI API call, defined by a prompt in the form of a list of messages and a response string"""
    n_message_tokens = ( num_gpt4_tokens_for_messages(messages) / 1000.0 )
    n_response_tokens = ( num_gpt4_tokens_for_response(response_string) / 1000.0 )
    
    if gpt_model == 'gpt-4-1106-preview':
        message_rate = float(config.get('chatgpt4_turbo_costs', 'prompt_per_thousand_tokens')) 
        response_rate = float(config.get('chatgpt4_turbo_costs', 'response_per_thousand_tokens'))
    # Default is gpt-4
    else:
        message_rate = float(config.get('chatgpt4_costs', 'prompt_per_thousand_tokens')) 
        response_rate = float(config.get('chatgpt4_costs', 'response_per_thousand_tokens'))
       
    input_cost = n_message_tokens * message_rate
    response_cost = n_response_tokens * response_rate
    return input_cost + response_cost
 
def num_gpt4_tokens_for_messages(messages):
  """Returns the number of tokens used by a list of messages.
Adapted from code at https://platform.openai.com/docs/guides/chat/introduction."""
  encoding = tiktoken.encoding_for_model("gpt-4")
  num_tokens = 0
  for message in messages:
      num_tokens += 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
      for key, value in message.items():
          num_tokens += len(encoding.encode(value))
          if key == "name":  # if there's a name, the role is omitted
              num_tokens += -1  # role is always required and always 1 token
  num_tokens += 2  # every reply is primed with <im_start>assistant
  return num_tokens

def num_gpt4_tokens_for_response(response_string):
  """Returns the number of tokens in a response."""
  encoding = tiktoken.encoding_for_model("gpt-4")
  return len(encoding.encode(response_string))

