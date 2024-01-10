from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import UnstructuredPDFLoader, PDFMinerLoader, PyMuPDFLoader
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate
import gdown
import os
from pymongo import MongoClient
from bson.objectid import ObjectId
from cleantext import clean
import chromadb
import torch
from dotenv import load_dotenv
load_dotenv()
k = 3


def response_to_structured(response: str):
    try:
        title_index = response.index('Title')
        summary_index = response.index('Summary')
        tags_index = response.index('Tags')
        title = response[title_index+7: summary_index]
        summary = response[summary_index+9: tags_index]
        tags = response[tags_index+6:]
        result = {
            'title': title.rstrip(),
            'summary': summary.rstrip(),
            'tags': tags.rstrip()
        }
        return result
    except:
        return {}


def init_keys(pdf_path, chunk_size=1000):
    loader = PyMuPDFLoader(pdf_path)
    data = loader.load()
    text_splitter = TokenTextSplitter(
        chunk_size=chunk_size, chunk_overlap=0, encoding_name='cl100k_base')
    texts = text_splitter.split_documents(data)
    k_first_texts = [chunks.page_content for chunks in texts[:k]]
    texts = ' '.join(text for text in k_first_texts)
    with open('model/prompts/init_doc_prompt.txt', 'r') as f:
        init_doc_prompt = f.readlines()
        init_doc_prompt = ''.join(x for x in init_doc_prompt)
    prompt = PromptTemplate(template=init_doc_prompt,
                            input_variables=['context'])
    chain = LLMChain(
        llm=ChatOpenAI(),
        prompt=prompt,
        verbose=True
    )
    result = chain.predict(context=texts)
    result_json = response_to_structured(result)
    return result_json
