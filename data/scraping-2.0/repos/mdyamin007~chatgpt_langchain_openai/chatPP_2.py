import os
import sys

import openai
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle
import faiss
from langchain.vectorstores import FAISS
# InstructorEmbedding 
from InstructorEmbedding import INSTRUCTOR
from langchain.embeddings import HuggingFaceInstructEmbeddings

import constants

os.environ["OPENAI_API_KEY"] = constants.APIKEY


def store_embeddings(docs, embeddings, store_name, path):
    
    vectorStore = FAISS.from_documents(docs, embeddings)

    with open(f"{path}/faiss_{store_name}.pkl", "wb") as f:
        pickle.dump(vectorStore, f)
        
def load_embeddings(store_name, path):
    with open(f"{path}/faiss_{store_name}.pkl", "rb") as f:
        VectorStore = pickle.load(f)
    return VectorStore

# Enable to save to disk & reuse the model (for repeated queries on the same data)
PERSIST = True

# query = sys.argv[1]
# embeddings = OpenAIEmbeddings()
embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
embedding_store_path = "embeddings"

if PERSIST and os.path.exists(embedding_store_path):
  print("Reusing index...\n")
  db = load_embeddings(store_name='openAIEmbeddings', 
                                    path=embedding_store_path)
else:
  #loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
  loader = DirectoryLoader("data/", silent_errors=True, show_progress=True, use_multithreading=True)
  documents = loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
  texts = text_splitter.split_documents(documents)
  if PERSIST:
    # os.mkdir(embedding_store_path)
    store_embeddings(texts, 
                  embeddings, 
                  store_name='openAIEmbeddings', 
                  path=embedding_store_path)
    db = load_embeddings(store_name='openAIEmbeddings', 
                                    path=embedding_store_path)
  else:
    db = FAISS.from_documents(texts, embeddings)
    

print("Welcome to the State of the ChatPP! Type 'exit' to stop.")
while True:
    query = input("Please enter your question: ")
    if query == "exit":
        break
    chain = RetrievalQA.from_chain_type(
      llm=ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0.9),
      chain_type="stuff",
      retriever=db.as_retriever(search_kwargs={"k": 3}),
    )
    print(chain.run(query))
