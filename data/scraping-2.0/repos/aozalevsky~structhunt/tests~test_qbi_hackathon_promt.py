#!/usr/bin/env python
# coding: utf-8

# In[9]:


import os
import re
import openai
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

import re
import requests
import time
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# OPENAI SETUP

openai_api_key = "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"
os.environ['OPENAI_API_KEY'] = openai_api_key


llm = ChatOpenAI(
            openai_api_key=openai_api_key,
            temperature=0, model_name="gpt-4", max_tokens=300, request_timeout = 30, max_retries=3
        )


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=100, separators=["\n\n", "\n", ".", ","]
    )

# text_splitter = CharacterTextSplitter(chunk_size=300, separators=["\n\n", "\n", ".", ","], chunk_overlap=100)


# In[10]:


keywords_groups = {
    'CX-MS': ['cross-link', 'crosslink', 'XL-MS', 'CX-MS', 'CL-MS', 'XLMS', 'CXMS', 'CLMS', "chemical crosslinking mass spectrometry", 'photo-crosslinking', 'crosslinking restraints', 'crosslinking-derived restraints', 'chemical crosslinking', 'in vivo crosslinking', 'crosslinking data'],
    'HDX': ['Hydrogen–deuterium exchange mass spectrometry', 'Hydrogen/deuterium exchange mass spectrometry' 'HDX', 'HDXMS', 'HDX-MS'],
    'EPR': ['electron paramagnetic resonance spectroscopy', 'EPR', 'DEER', "Double electron electron resonance spectroscopy"],
    'FRET': ['FRET',  "forster resonance energy transfer", "fluorescence resonance energy transfer"],
    'AFM': ['AFM',  "atomic force microscopy" ],
    'SAS': ['SAS', 'SAXS', 'SANS', "Small angle solution scattering", "solution scattering", "SEC-SAXS", "SEC-SAS", "SASBDB", "Small angle X-ray scattering", "Small angle neutron scattering"],
    '3DGENOME': ['HiC', 'Hi-C', "chromosome conformation capture"],
    'Y2H': ['Y2H', "yeast two-hybrid"],
    'DNA_FOOTPRINTING': ["DNA Footprinting", "hydroxyl radical footprinting"],
    'XRAY_TOMOGRAPHY': ["soft x-ray tomography"],
    'FTIR': ["FTIR", "Infrared spectroscopy", "Fourier-transform infrared spectroscopy"],
    'FLUORESCENCE': ["Fluorescence imaging", "fluorescence microscopy", "TIRF"], 
    'EVOLUTION': ['coevolution', "evolutionary covariance"],
    'PREDICTED': ["predicted contacts"],
    'INTEGRATIVE': ["integrative structure", "hybrid structure", "integrative modeling", "hybrid modeling"],
    'SHAPE': ['Hydroxyl Acylation analyzed by Primer Extension']
}


# In[11]:


methods_string = ''
for i, (k, v) in enumerate(keywords_groups.items()):
    if i > 0:
        methods_string += ' or '
    methods_string += f'{k} ({", ".join(v)})' 


# In[12]:


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

    print('text copied')

    loader = TextLoader(txtfilepath, autodetect_encoding=True)
    documents = loader.load()
    docs = text_splitter.split_documents(documents)

    faissIndex = FAISS.from_documents(docs, OpenAIEmbeddings())
    faissIndex.save_local(filepath)
    return FAISS.load_local(filepath, OpenAIEmbeddings())


def run_test(pmcid: str, queries: [str]):
    ## Write to file
    #pmcid = 'PMC9935389' 
    #pmcid = 'PMC10081221'
    text = get_pmc_paper(pmcid)
    methods_text = extract_methods_from_pmc_paper(text)
    with open('input_file.txt', 'w') as file:
        file.write(methods_text)


    loader = TextLoader("./input_file.txt")
    documents = loader.load()
    docs = text_splitter.split_documents(documents)
    
    print(docs)
    
    raise

    faissIndex = FAISS.from_documents(docs, OpenAIEmbeddings())
    current_document = "input_doc"
    faissIndex.save_local(current_document)
    #embeddings have been saved

    chatbot = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=c.load_local(current_document, OpenAIEmbeddings())
            .as_retriever(search_type="similarity", search_kwargs={"k": 1})
    )

    for q in queries:

        template = """ {query}? """

        prompt = PromptTemplate(
            input_variables=["query"],
            template=template,
        )

        print(chatbot.run(
            prompt.format(
                query = [f"You are reading a materials and methods section of a scientific paper. Here is the list of structural biology methods {methods_string}.\n\n Did the authors use any methods from the list? \n\n Answer with Yes or No followed by the names of the methods."],
                
        )))
        
        time.sleep(1)
        

    print('finished queries')


    # function that takes context, prompts, and returns answers 

    # comparison of answers to known answers

    # 

def fetch_embedding(pmcid):
    try:
        filepath = f'./data/{pmcid}'
        return FAISS.load_local(filepath, OpenAIEmbeddings())
    except Exception as e:
        print(e)
        print('failure')
        return embed_article(pmcid)

def result_to_bool(result):
    flag = False
    if re.match('Yes', result, re.IGNORECASE):
        flag = True
    return flag


def embed_all(pmc_ids):
    #pmc_ids = ['PMC8536336', 'PMC7417587', 'PMC5957504', 'PMC7492086', 'PMC9293004', 
    #'PMC7854634', 'PMC5648754', 'PMC8022279', 'PMC8655018', 'PMC9279154']
    y_test = []
    for pmcid in pmc_ids:
        embedding = fetch_embedding(pmcid)
        queries = [f"You are reading a materials and methods section of a scientific paper. Here is the list of methods {methods_string}.\n\n Did the authors use any of them? Answer Yes or No, followed by the name(s) of methods. Use only abbreviations."],
        
        result = evaluate_query(embedding, queries)
        
        has_yes = result_to_bool(result)
        if has_yes:
            val = True
            y_test.append(val)
        else:
            val = False 
            y_test.append(val)

        print(pmcid)
        print(result)
        print('-' * 80)
        time.sleep(5)
    return y_test

#y_pred = [True, True, True, True, True, True, True]
#y_test = [False, True, True, True, True, False, True]
def create_confusion(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, labels=[True, False])
    print(accuracy_score(y_test, y_pred))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['True', 'False'])
    disp.plot()
    plt.show()


#infile = "data.csv"
def get_pmcid_from_pmc_url(pmcurl):
    return re.split('/', pmcurl)[-1]

def load_file(infile):
    table = pd.read_excel(infile)
    b = table[~pd.isna(table['method_0'])].sample(n=10)
    c = table[pd.isna(table['method_0'])].sample(n=10)
    new_list = [b, c]
    #print(b)
    #print(c)
    links = pd.concat(new_list)
    pmc_ids = []
    y_test = []
    for index, row in links.iterrows():
        row_link = row['pmc']
        pmc_num = get_pmcid_from_pmc_url(row_link)
        pmc_ids.append(pmc_num)
        #print(row["method_0"])
        #print(not pd.isnull(row["method_0"]))
        #print(~pd.isna(row['method_0']))
        #print(not pd.isna(row['method_0']))
        
        if not pd.isna(row['method_0']):
            val = True
            y_test.append(val)
        else:
            val = False
            y_test.append(val)

    
    return y_test, pmc_ids


#create_confusion(y_pred, y_test)
#create_confusion(y_pred, y_test)
def evaluate_query(embedding, queries):
    chatbot = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=embedding.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    )
    
    template = """ {query}? """
    for q in queries:
        prompt = PromptTemplate(
            input_variables=["query"],
            template=template,
        )

        return(chatbot.run(
            prompt.format(query=q)
        ))    
infile = "data_qbi_all.xlsx"
y_test, pmcid_list = load_file(infile)
y_pred = embed_all(pmcid_list)
#print(y_pred)
#print(y_test)
create_confusion(y_test, y_pred)
#Accuracy 1-4.0: 0.8
#Accuracy 2-4.0: 0.85
#Accuracy 3-4.0: 0.8

#Average Accuracy --> 4.0: 0.82
#Standard Deviation --> 4.0: 0.024

#Accuracy 1-3.5: 0.6
#Accuracy 2-3.5: 0.7
#Accuracy 3-3.5: 0.55

#Average Accuracy --> 3.5: 0.62
#Standard Deviation --> 0.062





def main():
    #silly example
    queries = [f"Did the the authors use any methods from this list? {methods_string}. Answer with Yes or No followed by the name of the methods."]
               
    run_test(pmcid='PMC23402394802394', queries=queries)


# In[ ]:




