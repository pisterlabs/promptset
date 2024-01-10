import os
import getpass

from langchain.document_loaders import PyPDFLoader  #document loader: https://python.langchain.com/docs/modules/data_connection/document_loaders
from langchain.text_splitter import RecursiveCharacterTextSplitter  #document transformer: text splitter for chunking
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma #vector store
from langchain import HuggingFaceHub  #model hub
from langchain.chains import RetrievalQA
import chainlit as cl

#loading the API key
os.environ['HUGGINGFACEHUB_API_TOKEN'] = getpass.getpass('Hugging face api key:')

path = input("Enter PDF file path: ")
loader = PyPDFLoader(path)
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
docs = splitter.split_documents(pages)

embeddings = HuggingFaceEmbeddings()
doc_search = Chroma.from_documents(docs, embeddings)

repo_id = "tiiuae/falcon-7b"
llm = HuggingFaceHub(repo_id=repo_id, model_kwargs={'temperature': 0.2, 'max_length':1000}) 

@cl.on_chat_start
def main():
    retrieval_chain = RetrievalQA.from_chain_type(llm, chain_type='stuff', retriever=doc_search.as_retriever())
    cl.user_session.set("retrieval_chain", retrieval_chain)
    
@cl.on_message
async def main(message:str):
    retrieval_chain = cl.user_session.get("retrieval_chain")
    res = await retrieval_chain.acall(message, callbacks=
                                      [cl.AsyncLangchainCallbackHandler()])
    
    #print(res)
    await cl.Message(content=res["result"]).send()