import sys
import fire
import logging

import os
if os.environ.get("OPENAI_API_KEY") == "":
  print("`OPENAI_API_KEY` is not set", file=sys.stderr)
  sys.exit(1)

url = os.environ.get("WORDPRESS_URL")
username = os.environ.get("WORDPRESS_USERNAME")
password = os.environ.get("WORDPRESS_PASSWORD")

from llama_index import GPTSimpleVectorIndex, download_loader
from llama_index.langchain_helpers.chatgpt import ChatGPTLLMPredictor

from llama_index import GPTSimpleVectorIndex, LLMPredictor
from langchain.llms import OpenAI

prompt = """
Answer the question below based on the blog's content as if you were the author.

Question: {question}

Answer to the question in the same language as the question.
"""

def run(make_index=False, query=False, top_k=5, verbose=False):
  if verbose:
    logging.basicConfig(stream=sys.stdout, level=logging.DEBUG, force=True)
  else:
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING, force=True)

  if make_index:
    do_make_index()
  elif query != False:
    do_query(query, top_k)
  else:
    do_chat(top_k)

def load_index():
  llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, max_tokens=1024))
  return GPTSimpleVectorIndex.load_from_disk(
    "data/wordpress.json",
    llm_predictor=llm_predictor,
  )

def do_make_index():
  WordpressReader = download_loader("WordpressReader")

  loader = WordpressReader(url=url, username=username, password=password)
  documents = loader.load_data()

  index = GPTSimpleVectorIndex(documents, llm_predictor=ChatGPTLLMPredictor())
  index.save_to_disk("data/wordpress.json")

def execute_query(index, query, top_k):
  output = index.query(prompt.format(question=query), similarity_top_k=top_k)
  return output

def do_query(query, top_k):
  index = load_index()
  output = execute_query(index, prompt.format(question=query), top_k)
  print(output)

def do_chat(top_k):
  print("Loading index...")
  index = load_index()

  print("Question: ", end="", flush=True)
  try:
    while question := next(sys.stdin).strip():
      output = execute_query(index, prompt.format(question=question), top_k)
      print("Answer: ", end="")
      print(output)
      print("")
      print("Question: ", end="", flush=True)
  except KeyboardInterrupt:
    pass

if __name__ == '__main__':
  fire.Fire(run)
