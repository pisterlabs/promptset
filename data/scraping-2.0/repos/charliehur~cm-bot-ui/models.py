import os

import streamlit as st

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Pinecone
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import PyPDFLoader,JSONLoader
import pinecone as pinecone
import pandas as pd
import random


@st.cache_resource
def load_retriever():
    data = []


    data.extend(CSVLoader(file_path="articles.csv", encoding='ISO-8859-1').load_and_split())
    data.extend(CSVLoader(file_path="cooking_terms.csv", encoding='ISO-8859-1').load_and_split())
    #data.extend(PyPDFLoader(file_path="lyrics.pdf").load_and_split())
    #data.extend(JSONLoader(file_path="glossary.json",).load_and_split())

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY'])
    # initialize pinecone
    pinecone.init(
        api_key=os.environ['PINECONE_API_KEY'],  # find at app.pinecone.io
        environment=os.environ['PINECONE_API_ENV']  # next to api key in console
    )

    metadata_field_info = [
        AttributeInfo(
            name="Cooking_Term",
            description="Term used by the chef in the dining hall of the Colonize Mars colony",
            type="string"
        ),
        AttributeInfo(
            name="Part_of_Speech",
            description="English part of speech for the cooking term (noun, verb, adjective, adverb, etc.)",
            type="string"
        ),
        AttributeInfo(
            name="Definition",
            description="Meaning of the cooking term in the context of the Colonize Mars colony",
            type="string"
        )]
    
    index_name = os.environ['PINECONE_INDEX_NAME'] # put in the name of your pinecone index here

    #vector_store = Pinecone.from_existing_index(index_name=os.environ["PINECONE_INDEX_NAME"], embedding=OpenAIEmbeddings())
    vector_store = Pinecone.from_documents(data, embeddings, index_name=index_name)
    llm = OpenAI(model_name="gpt-4", temperature=0.7)
    document_content_description = "Colonize Mars user guide and cooking terms"

    return SelfQueryRetriever.from_llm(
        llm, vector_store, document_content_description, metadata_field_info
    )    

def load_chain():

    #Theme of song should be result of analogy: rocketship:{topic} = {question}:_______
    template = """Context information is below.
    ---------------------
    {context}
    ---------------------
    You are the chef in the dining hall of the Colonize Mars colony. Given the context information, 
    write a tweet about {question} and in the answer, use one and only one cooking term defined in the context information. 
    Answer:
    """

    prompt = PromptTemplate(
        input_variables=["question", "context"], 
        template=template
    )

    llm = OpenAI(model_name="gpt-4", temperature=0.7)
    #llm.logit_bias = {"8237":-100, "47553": -100}

    return load_qa_chain(llm, verbose=True, chain_type="stuff", prompt=prompt)

context_search = load_retriever()
chain = load_chain()
