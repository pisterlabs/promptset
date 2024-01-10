import os
import openai
from transformers import GPT2TokenizerFast

_TOKENIZER = GPT2TokenizerFast.from_pretrained('gpt2')
GPT3_LENGTH_LIMIT = 2049
openai.api_key = os.getenv("OPENAI_API_KEY")

def gpt_style_tokenize(x):
    return _TOKENIZER.tokenize(x)

def length_of_prompt(prompt, max_tokens):
    return len(_TOKENIZER.tokenize(prompt)) + max_tokens
