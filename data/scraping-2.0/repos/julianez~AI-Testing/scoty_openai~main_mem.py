import os
import logging
from langchain.prompts.chat import (
    PromptTemplate
)
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain.llms import AzureOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import Qdrant
from langchain.output_parsers import PydanticOutputParser
from qdrant_client.http import models as rest
from pydantic import BaseModel, Field
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders import WebBaseLoader


from typing import Dict, List, Optional
from langchain.document_loaders.base import BaseLoader
from langchain.docstore.document import Document


with open('URLs-Scotiabank-CL.txt', 'r') as f:
     lines = f.readlines()

urls = [line.strip() for line in lines]

loader = WebBaseLoader(urls)
webDocs = loader.load()

embeddings = AzureOpenAIEmbeddings()

quadrant_docsearch = Qdrant.from_documents(
    webDocs,
    embeddings,
    location=":memory:",
    collection_name="book"
)


openaiLLM = AzureOpenAI(
    azure_endpoint="https://openai-jez.openai.azure.com/",
    azure_deployment="GPT-35-turbo",
    model="GPT-35-turbo",
    api_version="2023-05-15"
)


qa = RetrievalQA.from_chain_type(
    llm=openaiLLM, chain_type="stuff", retriever=quadrant_docsearch.as_retriever(), return_source_documents=True)

while True:
    user_input = input("Hola, soy Scoty en que puedo ayudarte?\n")

    help_request = "Eres un asesor. Ayuda al usuario. Si no sabes la respuesta di que no tienes la información." +\
        f"\nUser:{user_input}"
    result = qa({"query": help_request})
    #print(len(result['source_documents']))
    print(result["result"])
