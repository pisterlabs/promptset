

import os
import openai
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from openai.error import Timeout

import re
import requests
import xml.etree.ElementTree as ET


from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


# OPENAI SETUP

openai_api_key = "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"
os.environ['OPENAI_API_KEY'] = openai_api_key

class LlmHandler:

    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(separators = ["\n\n", '\n', '.'], chunk_size=1000, chunk_overlap=0)
        self.llm=ChatOpenAI(
                openai_api_key=openai_api_key,
                temperature=0, model_name="gpt-3.5-turbo", max_tokens=500, request_timeout = 15, max_retries=1
            )
        
        
    def evaluate_queries(self, embedding, queries):
        chatbot = RetrievalQA.from_chain_type(
            llm=self.llm, 
            chain_type="stuff", 
            retriever=embedding.as_retriever(search_type="similarity", search_kwargs={"k":1})
        )
        
        template = """ {query}? """
        response = []
        for q in queries:
            prompt = PromptTemplate(
                input_variables=["query"],
                template=template,
            )

            response.append(chatbot.run(
                prompt.format(query=q)
            ))
        return response


def get_pmc_paper(pmcid):
    url = f'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML'
    req = requests.get(url)
    res = req.text
    return res

def extract_methods_from_pmc_paper(paper):
    tree = ET.fromstring(paper)

    mtext = []
    for sec in tree.iter('sec'):
        for title in sec.iter('title'):
            if isinstance(title.text, str):
                if re.search('methods', title.text, re.IGNORECASE):
                    mtext.extend(list(sec.itertext()))

    return " ".join(mtext)

def preprocess(input_text):
    processed_data = input_text.replace("\n","")
    return processed_data

def embed_article(pmcid):
    text = get_pmc_paper(pmcid)
    methods_text = preprocess(extract_methods_from_pmc_paper(text))
    filepath = f'./data/{pmcid}'
    txtfilepath = filepath + '.txt'
    with open(txtfilepath, 'w', encoding="utf-8") as file:
        file.write(methods_text)

    loader = TextLoader(txtfilepath, autodetect_encoding=True)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(separators = ["\n\n", '\n', '.'], chunk_size=1000, chunk_overlap=0)
        
    docs = text_splitter.split_documents(documents)

    faissIndex = FAISS.from_documents(docs, OpenAIEmbeddings())
    faissIndex.save_local(filepath)
    return FAISS.load_local(filepath, OpenAIEmbeddings())

def fetch_embedding(pmcid, bypass_cache=False):
    if bypass_cache:
        return embed_article(pmcid)
    try:
        filepath = f'./data/{pmcid}'
        return FAISS.load_local(filepath, OpenAIEmbeddings())
    except Exception as e:
        print("failed to load cached embedding, generating embedding")
        return embed_article(pmcid)


def compare_against_known():
    # only asking "are there more methodologies beyond "
    queries = ["Are there experimental techniques beyond using Cryo-Em incorporated in the paper? Answer with Yes or No followed by the experimental technique."]
    pmc_ids_false = ['PMC8536336', 'PMC7417587', 'PMC5957504', 'PMC7492086', 'PMC9293004']
    pmc_ids_true = ['PMC7854634', 'PMC5648754', 'PMC8022279', 'PMC8655018', 'PMC8916737']

    for pmc in pmc_ids_false:
        #run_test(pmc, )
        pass

queries = ["Are there experimental techniques beyond using Cryo-Em incorporated in the paper? Answer with Yes or No followed by the experimental technique."]

def run_all_tests(handler):
    pmc_ids = ['PMC8536336', 'PMC7417587', 'PMC5957504', 'PMC7492086', 'PMC9293004', 
    'PMC7854634', 'PMC5648754', 'PMC8022279', 'PMC8655018', 'PMC8916737']
    for pmcid in pmc_ids:
        embedding = fetch_embedding(pmcid)
        print(pmcid)
        result = evaluate_query(embedding, queries)
        print(result)


queries = ["Are there experimental techniques incorporated in the paper? Do not incorporate Cryo-EM related to structural biology or integrative modeling?"]

def evaluate_query(embedding, queries):
    chatbot = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(
            openai_api_key=openai_api_key,
            temperature=0, model_name="gpt-4", max_tokens=50, request_timeout = 20, max_retries=1
        ), 
        chain_type="stuff", 
        retriever=embedding.as_retriever(search_type="similarity", search_kwargs={"k":1})

    )

    

handler = LlmHandler()
run_all_tests(handler)

