
import os

import constants
import sys
from langchain.document_loaders import TextLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain import HuggingFaceHub
from langchain.vectorstores import Chroma

# Importamos las apikeys con os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = constants.HUGGINGFACEHUB_API_TOKEN

PERSIST = False

query = None
if len(sys.argv) > 1:
  query = sys.argv[1]

if PERSIST and os.path.exists("persist"):
  print("Reusing index...\n")
  vectorstore = Chroma(persist_directory="persist", embedding_function=HuggingFaceEmbeddings())
  index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
  loader = TextLoader("data/data.txt") 
  if PERSIST:
    index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
  else:
    index = VectorstoreIndexCreator(embedding=HuggingFaceEmbeddings()).from_loaders([loader])

llm = HuggingFaceHub(repo_id="declare-lab/flan-alpaca-large", model_kwargs={"temperature": 0.1,"max_length": 64})

chain = ConversationalRetrievalChain.from_llm(
  llm=llm,
  retriever=index.vectorstore.as_retriever(search_kwargs={"k": 1}),
)


chat_history = []
while True:
  if not query:
    query = input("Prompt: ")
  if query in ['quit', 'q', 'exit']:
    sys.exit()
  result = chain({"question": query, "chat_history": chat_history})
  print(result['answer'], len(result["answer"]))

  chat_history.append((query, result['answer']))
  query = None
