import os
import openai
from dotenv import load_dotenv
import tiktoken

import sys

load_dotenv()
prompt = sys.stdin.read()

openai.api_key = os.getenv("OPENAI_API_KEY")


def open_ai_get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0,
       )
    except openai.error.InvalidRequestError as e:
         print("An error occurred while trying to get completion from OpenAI. prompt word count ")
         print(len(prompt.split()) )
         print(e)
         sys.exit(1)
    return response.choices[0].message["content"]

def split_large_text(large_text, max_tokens):
  """Split a large text into chunks with a token limit.

  Args:
    large_text: The large text to split.
    max_tokens: The maximum number of tokens per chunk.

  Returns:
    A list of text chunks.
  """

  encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
  tokenized_text = encoding.encode(large_text)

  sofar = 0 
  chunks = []
  while (sofar < len(tokenized_text)):
    chunk1=encoding.decode(tokenized_text[sofar:sofar+max_tokens])
    print(chunk1)
    sofar += max_tokens
    chunks.append(chunk1)
  return chunks


max_tokens = 4000

chunks = split_large_text(prompt, max_tokens)

x="Please summarize the following text in a few bullet points: \n"

for chunk in chunks:
    print(open_ai_get_completion(x + chunk))
    print ("------------------")