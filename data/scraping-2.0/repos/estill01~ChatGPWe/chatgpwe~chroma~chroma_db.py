import os
from typing import List, Dict, Union

import openai
import tiktoken
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction


def build_prompt(query: str, context: List[str]) -> List[Dict[str, str]]:
  """
  Builds a prompt for the LLM. # 
  
  This function builds a prompt for the LLM. It takes the original query,
  and the returned context, and asks the model to answer the question based only
  on what's in the context, not what's in its weights. 

  More information: https://platform.openai.com/docs/guides/chat/introduction

  Args:
  query (str): The original query.
  context (List[str]): The context of the query, returned by embedding search.

  Returns:
  A prompt for the LLM (List[Dict[str, str]]).
  """

  system = {
    'role':
    'system',
    'content':
    'I am going to ask you a question, which I would like you to answer based only on the provided context, and not any other information. If there is not enough information in the context to answer the question, say "I am not sure", then try to make a guess. Break your answer up into nicely readable paragraphs.'
  }
  user = {
    'role':
    'user',
    'content':
    f"""
           The question is {query}. Here is all the context you have: {(' ').join(context)}
           """
  }

  return [system, user]


def get_chatGPT_response(query: str, context: List[str]) -> str:
  """
  Queries the GPT API to get a response to the question.

  Args:
  query (str): The original query.
  context (List[str]): The context of the query, returned by embedding search.

  Returns:
  A response to the question.
  """

  response = openai.ChatCompletion.create(
    model='gpt-3.5-turbo',
    messages=build_prompt(query, context),
  )

  return response.choices[0].message.content


def filter_results(
  results: Dict[str, List[Union[Dict[str, str], str]]],
  max_prompt_length: int = 3900
) -> Dict[str, List[Union[Dict[str, str], str]]]:
  """
  Filters the query results so they don't exceed the model's token limit. 

  Args:
  results (Dict[str, List[Union[Dict[str, str], str]]]): The query results.
  max_prompt_length (int): The maximum length of the prompt.

  Returns:
  The filtered query results.
  """

  contexts = []
  sources = []

  # We use the tokenizer from the same model to get a token count
  tokenizer = tiktoken.encoding_for_model('gpt-3.5-turbo')

  total = 0
  for i, c in enumerate(results['documents'][0]):
    total += len(tokenizer.encode(c))
    if total <= max_prompt_length:
      contexts.append(c)
      sources.append(results['metadatas'][0][i]['page_number'])
    else:
      break
  return contexts, sources


def get_collection(collection_name='russel_norvig',
                   persist_directory='chroma-russel-norvig'):
  """
  Instantiates the Chroma client, and returns the collection.
  """

  # Instantiate the Chroma client. We use persistence to load the already existing index.
  # Learn more at docs.trychroma.com
  client = chromadb.Client(settings=Settings(
    chroma_db_impl="duckdb+parquet", persist_directory=persist_directory))

  # Get the collection. We use the same embedding function we used to create the index.
  embedding_function = OpenAIEmbeddingFunction(
    api_key=os.environ['OPENAI_API_KEY'])
  collection = client.get_or_create_collection(name=collection_name,
                                     embedding_function=embedding_function)

  return collection


def main():
  # We set the OpenAI API Key since we'll also call their APIs directly
  # Be sure to have this set in `secrets`.

  # Check if the OPENAI_API_KEY environment variable is set. If not, exit.
  if 'OPENAI_API_KEY' not in os.environ:
    print(
      "You need to set your OpenAI API key as the OPENAI_API_KEY environment variable in Secrets. Get your key from https://platform.openai.com/account/api-keys"
    )
    return 0

  openai.api_key = os.environ['OPENAI_API_KEY']

  # Get the chroma collection
  collection = get_collection()

  print("""
This is a demo demonstrating how to plug knowledge into LLMs using Chroma.

We've turned the popular AI textbook, "Artificial Intelligence: A Modern Approach" by Stuart Russell and Peter Norvig, into pluggable knowledge - this demo lets you ask questions about the content of the book, and get answers with page references. 

Input a Query to get answers. Some queries to try:
- What is an intelligent agent?
- What are backtracking search algorithms?
- What can you tell me about machine learning?
- What is the difference between a computer and a robot?
  
  """)

  # We use a simple input loop.
  while True:

    # Get the user's query
    query = input("Query: ")
    if len(query) == 0:
      print("Please enter a question. Ctrl+C to Quit.\n")
      continue
    print('\nThinking...\n')

    # Query the collection to get the 5 most relevant results
    results = collection.query(query_texts=[query],
                               n_results=5,
                               include=['documents', 'metadatas'])

    MAX_PROMPT_LENGTH = 3900  #The maximum number of tokens gpt-3.5-turbo can use as a prompt, with some room for the fixed part of the prompt.
    # Filter the query results so they fit within the max prompt length.
    contexts, sources = filter_results(results, MAX_PROMPT_LENGTH)

    # Get the response from GPT
    response = get_chatGPT_response(query, contexts)

    # Output, with sources
    print(response)
    print('\n')
    print(f"Source pages: {sources}")
    print('\n')


if __name__ == '__main__':
  main()
