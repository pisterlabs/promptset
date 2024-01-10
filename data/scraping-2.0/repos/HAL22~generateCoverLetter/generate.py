import os
import streamlit as st
import pinecone
import string
import random
from langchain.llms import OpenAI
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Pinecone
from langchain.chains import LLMChain

pine_cone_name = "coverletter"

def fill_keys(openAIKey,pineconeAPIKey,pineconeEnv):
    os.environ['OPENAI_API_KEY'] = openAIKey
    os.environ['PINECONE_API_KEY'] = pineconeAPIKey
    os.environ['PINECONE_ENV'] = pineconeEnv

def get_index(filename):
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    pinecone.init(
    api_key=os.getenv('PINECONE_API_KEY'),
    environment=os.getenv('PINECONE_ENV')
    )

    pinecone.create_index(pine_cone_name, dimension=1536)
    loader = PyPDFLoader(filename)
    pages = loader.load_and_split()   

    return Pinecone.from_documents(pages, embeddings, index_name=pine_cone_name) 

def generate_cover_letter(index,name,temp=0.1):
    prompt_template = """Use the context below to write a cover letter:
    Context: {context}
    Cover letter:"""
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context"])
    llm = OpenAI(temperature=temp, verbose=True)
    chain = LLMChain(llm=llm, prompt=PROMPT)

    docs = index.similarity_search(name, k=4)
    inputs = [{"context": doc.page_content} for doc in docs]  
    letter = chain.apply(inputs)
    
    pinecone.delete_index(pine_cone_name)
    return letter[0]["text"]