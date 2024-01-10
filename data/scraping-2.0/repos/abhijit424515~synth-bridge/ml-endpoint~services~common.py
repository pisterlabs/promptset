import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.vectorstores.chroma import Chroma
from dotenv import load_dotenv
import chromadb

TTL = 900
load_dotenv("../.env")
openai_api_key = os.environ.get("OPENAI_API_KEY")
redis_conn = os.environ.get("REDIS")

# Instance of ChromaDB vector store for context-based chat interactions
client = chromadb.PersistentClient(path="./kt-db")

# Instance of OpenAI's chat function
model = ChatOpenAI(
   model_name = "gpt-4",
   openai_api_key = os.environ.get("OPENAI_API_KEY")
)

# Instance of OpenAI's embeddings function
embed = OpenAIEmbeddings(
  model='text-embedding-ada-002',
  openai_api_key=openai_api_key
)

# Retrieve project data as context, and make a query on it
def context_chat(id, prompt, query):
  retriever = Chroma(
      client=client,
      collection_name=id,
      embedding_function=embed,
    ).as_retriever()
  prompt = ChatPromptTemplate.from_template(prompt)
  output = (
        {"context": retriever, "query": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    ).invoke(query)
  return output