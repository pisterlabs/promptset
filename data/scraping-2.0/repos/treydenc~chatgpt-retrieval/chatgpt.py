import os
import sys

import openai
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = False

query = None
# if len(sys.argv) > 1:
#   query = sys.argv[1]

if PERSIST and os.path.exists("persist"):
  print("Reusing index...\n")
  vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
  #loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
  loader = DirectoryLoader("data/")
  if PERSIST:
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
  else:
    index = VectorstoreIndexCreator().from_loaders([loader])

chain = ConversationalRetrievalChain.from_llm(
  llm=ChatOpenAI(model="gpt-3.5-turbo", temperature=0.6),
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)

# scene_creation_prompt = """
# These are the information of a person who has been going through something. Based on these information, generate some fantasy elements
# scenarios/acting scenes. The scenes are about the person experiencing anger while going through the certain things.
# Make up 5 stages, each with more complexity. Each stage has a different story.
# Make each story separate in time and space from the previous one. So they're new but with the same background. Describe the story. 
# The feeling of anger goes from simple to understand to hard to understand.
# It has to be in the following format:
# Stage 1: 
# Stage 2: 
# Stage 3:
# Stage 4: 
# Stage 5: 
# Do it only in English. 
# """

scene_creation_prompt = """
These are the information of a person who has been going through something. 
With these information, create a fantasy story with 5 distinguishable scenes that are a process of whatever the person is going through.
Make up 5 stages, each with more complexity. Each stage has a different story.
Make each story separate in time and space from the previous one. So they're new but with the same background. Describe the story. 
The feeling of anger goes from simple to understand to hard to understand.
It has to be in the following format:
Stage 1: 
Stage 2: 
Stage 3:
Stage 4: 
Stage 5: 
Do it only in English. 
"""

questions = ["What is the problem of the person? ", "What are they feeling?", "Why are they feeling it?"]
chat_history = []
counter = 0
while True:
  if counter < len(questions):
    query = questions[counter]
  
  if counter == len(questions):
    query = scene_creation_prompt
  # if not query:
  #   query = input("Prompt: ")
    # query =
  # if query in ['quit', 'q', 'exit']:
  #       sys.exit()
  if counter > len(questions):
    sys.exit()
  
  result = chain({"question": query, "chat_history": chat_history})
  print(result['answer'])

  chat_history.append((query, result['answer']))
  query = None
  counter += 1
