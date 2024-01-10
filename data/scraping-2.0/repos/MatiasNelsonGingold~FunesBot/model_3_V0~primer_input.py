
from haystack.document_stores import PineconeDocumentStore
from langchain.document_loaders import PyPDFLoader
from params import *
from primer_input import *
import pandas as pd


def primer_input_func():

    #Initializing the PineconeDocumentStore
    document_store = PineconeDocumentStore(
        api_key="ead8051e-8608-497a-beb2-a8d7d9092bd4",
        environment='asia-southeast1-gcp-free',
        similarity="dot_product",
        index='haystack-extractive-qa',
        embedding_dim=384
    )

    return document_store

path_pdf = './War and Peace (Leo Tolstoy).pdf'

#Lectura y extracción del contenido del PDF
loader = PyPDFLoader(path_pdf)
pages = loader.load_and_split()
book = [page.page_content for page in pages]

meta_datos = pd.read_csv('war_and_peace.csv')
