from pymongo import MongoClient
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import MongoDBAtlastVectorSearch
from langchain.document_loaders import DirectoryLoader
from langchain.llms import OpenAI 
from langchain.chains import RetrievalQA
import gradio as gr 
from gradio.themes.base import Base
import key_param

client = MongoClient(key_param.MONGO_URI)
dbName = "langchain_demo"
collectionName = "collection_of_text_blobs"
collection = client[dbName][collectionName]

embeddings = OpenAIEmbeddings(openai_api_type=key_param.openai_api_key)

vectorStore = MongoDBAtlastVectorSearch.from_documents(collection, embeddings)

def query_data(query):
  docs = vectorStore.similarity_search(query, K=1)
  as_output = docs[0].page_content

  llm = OpenAI(openai_api_key=key_param.openai_api_key, temperature = 0)
  retriever = vectorStore.as_retriever()
  qa = RetrievalQA.from_chain_type(llm, chain_type='stuff', retriever=retriever)
  retriever_output = qa.run(query)

  return as_output, retriever_output

# with gr.Blocks(theme=Base(), title='question ansering app using vector search + rag') as demo:


