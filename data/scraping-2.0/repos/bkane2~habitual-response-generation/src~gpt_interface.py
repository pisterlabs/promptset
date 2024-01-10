import backoff
import openai
from openai.error import RateLimitError, Timeout, ServiceUnavailableError, APIConnectionError, APIError
from transformers import GPT2Tokenizer

from util import *

openai.api_key = read_file('_keys/openai.txt')

TOKENIZER = GPT2Tokenizer.from_pretrained("gpt2")

MODEL_COSTS = {
  'text-davinci-003' : 0.02,
  'gpt-3.5-turbo' : 0.002
}

AVG_TOKENS_PER_CHAR = 0.25
  


@backoff.on_exception(backoff.expo, (RateLimitError, Timeout, ServiceUnavailableError, APIConnectionError, APIError))
def generate_instruct_gpt(model, prompt, suffix=None, stop=None, max_tokens=256):
  response = openai.Completion.create(
    model=model,
    prompt=prompt,
    suffix=suffix,
    max_tokens=max_tokens,
    stop=stop
  )
  return response



def cost_instruct_gpt(model, prompt, avg_resp_len, suffix=None, stop=None, max_tokens=256, tokenizer=TOKENIZER):
  tokens = tokenizer(prompt)['input_ids']
  n_tokens = len(tokens)
  if suffix:
    n_tokens += len(tokenizer(suffix)['input_ids'])
  n_tokens += AVG_TOKENS_PER_CHAR * min(avg_resp_len, max_tokens)
  return (MODEL_COSTS[model] / 1000) * n_tokens



@backoff.on_exception(backoff.expo, (RateLimitError, Timeout, ServiceUnavailableError, APIConnectionError, APIError))
def generate_chat_gpt(model, prompt, preamble=None, examples=[], stop=None, max_tokens=2048):
  messages=[]
  if preamble:
    messages.append({"role": "system", "content": preamble})
  for example in examples:
    messages.append({"role": "user", "content": example[0]})
    messages.append({"role": "assistant", "content": example[1]})
  messages.append({"role": "user", "content": prompt})

  response = openai.ChatCompletion.create(
    model=model,
    messages=messages,
    stop=stop,
    max_tokens=max_tokens
  )
  return response



def cost_chat_gpt(model, prompt, avg_resp_len, preamble=None, examples=[], stop=None, max_tokens=1024, tokenizer=TOKENIZER):
  n_tokens = 0
  if preamble:
    n_tokens += len(tokenizer(preamble)['input_ids'])
  for example in examples:
    n_tokens += len(tokenizer(example[0])['input_ids'])
    n_tokens += len(tokenizer(example[1])['input_ids'])
  n_tokens += len(tokenizer(prompt)['input_ids'])
  n_tokens += AVG_TOKENS_PER_CHAR * min(avg_resp_len, max_tokens)
  return (MODEL_COSTS[model] / 1000) * n_tokens