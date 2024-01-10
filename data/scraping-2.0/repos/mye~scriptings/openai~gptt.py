import sys
import os

import openai
from openai.error import APIError
from dotenv import load_dotenv

load_dotenv()

assert len(sys.argv) >= 5, "Must call with model_name context_length temperature max_tok [stop, ...]"
_, model_name, context_length, temperature, max_tokens, *stop_words = sys.argv
context_length, temperature, max_tokens = int(context_length), float(temperature), int(max_tokens)

assert len(stop_words) <= 4, "Can't provide more than 4 stop sequences"

api_key = os.getenv('OPENAI_API_KEY')
assert api_key
openai.api_key = api_key

# about 4 characters per token
# XXX: how many chars should be read?
# Some endpoints have a shared maximum of 2048 tokens
# prompt tokens plus max_tokens cannot exceed the model's context length
# don't read prompts longer than half the context length
# XXX: print warning if input was truncated

prompt = sys.stdin.read(context_length * 4) # should be upper bound for chars to read

# The max_tokens provided as an argument is easier to estimate as the maximal
# number of tokens for the answer, excluding the prompt
max_answer_tokens = len(prompt) // 4 + max_tokens

assert max_answer_tokens <= context_length, "prompt length + answer length don't fit in the models context"

try:
    response = openai.Completion.create(
        model=model_name,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_answer_tokens,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
except APIError as e:
    sys.stdout.close()
    raise e
else:
    print(response['choices'][0]['text'])
