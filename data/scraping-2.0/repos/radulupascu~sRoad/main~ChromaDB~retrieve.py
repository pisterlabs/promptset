import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA



def similarity_search(query, API_KEY, path, k=2):
  # Now we can load the persisted database from disk, and use it as normal. 
  persist_directory = path

  ## here we are using OpenAI embeddings but in future we will swap out to local embeddings
  embedding = OpenAIEmbeddings(openai_api_key=API_KEY)
  vectordb = Chroma(persist_directory=persist_directory, 
                  embedding_function=embedding)

  result = vectordb.similarity_search(query, k)
  print(result)
  return result

# def retriever(query):
#   retriever = vectordb.as_retriever(search_kwargs={"k": 2})
#   docs = retriever.get_relevant_documents(query)
#   return docs

if __name__ == "__main__":
  try:
    API_KEY = open("../API_KEY", "r").read()
  except FileNotFoundError:
    API_KEY = open("API_KEY", "r").read()

  persist_directory = '../db'

  ## here we are using OpenAI embeddings but in future we will swap out to local embeddings
  embedding = OpenAIEmbeddings(openai_api_key=API_KEY)
  vectordb = Chroma(persist_directory=persist_directory, 
                    embedding_function=embedding)
  while True:
    user_message = input("ASK: ") # Replace with input from website
    if user_message.lower() == "quit":
      break
    else:
      response = similarity_search(user_message, API_KEY, persist_directory, k=10)
      print(response)

