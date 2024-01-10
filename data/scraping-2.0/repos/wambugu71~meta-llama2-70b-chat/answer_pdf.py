#from langchain.embeddings import HuggingFaceHubEmbeddings
#1from langchain.document_loaders import TextLoader
#from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceHubEmbeddings
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
import os
#import warnings
import random
import string
HUG_TOKEN= os.environ["HUGGINGFACEHUB_API_TOKEN"] 
repo_id = "sentence-transformers/all-mpnet-base-v2"

def question_pdf(llm, text, prompt):
    token = os.environ["HUGGINGFACEHUB_API_TOKEN"]
    os.environ["HUGGINGFACEHUB_API_TOKEN"] 
    repo_id = "sentence-transformers/all-mpnet-base-v2"
    embeddings = HuggingFaceHubEmbeddings(
                repo_id=repo_id,
                task="feature-extraction"
            )
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents(text)
    
    db = Chroma.from_documents(texts, embeddings)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    #qa("what is data quality")
    return  qa({"query": f"{prompt}"})#['result']
    ####
    
