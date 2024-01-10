# %% [markdown]
# 

# %% [markdown]
# # Dataset FinanceBench

# %%
# !pip install qdrant-client

# %%
import pandas as pd
import os
import requests
from datasets import load_dataset
from datasets import DatasetDict
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import DirectoryLoader
import time

import sentence_transformers
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

# Load OpenAI access
import sys
sys.path.append(os.path.abspath('../../src'))
from azure_openai_conn import OpenAIembeddings

# %%
# Turn huggingface dataset to pd
# images = fashion["image"]
# data = fashion.remove_columns("image")
# product_df = data.to_pandas()
# product_data = product_df.reset_index(drop=True).to_dict(orient="index")
if os.path.isfile('../../data/financebench_sample_150.csv'):
    df = pd.read_csv('../../data/financebench_sample_150.csv')
else:    
    ds = load_dataset("PatronusAI/financebench")
    df = pd.DataFrame(ds)
    all_dicts = []
    for index, row in df.iterrows():    
        dictionary = row['train']    
        all_dicts.append(dictionary)
    df = pd.DataFrame(all_dicts)

# %%

destination_folder = '../../data/financebench'

if not os.path.exists(destination_folder):

    os.makedirs(destination_folder)

    for index, row in df.iterrows():
        url = row['doc_link']
        doc_name = row['doc_name']
        doc_name_with_extension = doc_name + '.pdf'        
        file_path = os.path.join(destination_folder, doc_name_with_extension)
        response = requests.get(url)
        if response.status_code == 200:            
            with open(file_path, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded: {doc_name_with_extension}")
        else:
            print(f"Failed to download: {doc_name_with_extension} ({url})")


# %%
pdf_folder_path = destination_folder
documents = []
for file in os.listdir(pdf_folder_path)[:5]:
    print(file)
    if file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder_path, file)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

# %%
len(documents)

# %% [markdown]
# Intuition about chunk-size: 
# GPT-3.5-turbo supports a context window of 4096 tokens — that means that input tokens + generated ( / completion) output tokens, cannot total more than 4096 without hitting an error. So we 100% need to keep below this. If we assume a very safe margin of ~2000 tokens for the input prompt into gpt-3.5-turbo, leaving ~2000 tokens for conversation history and completion. With this ~2000 token limit we can include 
# - 5 snippets of relevant information, meaning each snippet can be no more than 400 token long, or
# - 4 x 500
# 
# 

# %%
embeddings = OpenAIembeddings()

# %%
# Some grid-search
pchunks_list = [400, 600, 800, 1000]
poverlap_list = [30, 50, 80]

# Loop over each combination of chunk_size and chunk_overlap
for chunk_size in pchunks_list:
    for chunk_overlap in poverlap_list:
        try:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
            chunked_documents = text_splitter.split_documents(documents)
            
            # Create a unique directory name for each combination
            persist_directory = f'chroma_{chunk_size}_{chunk_overlap}'
            
            chroma = Chroma.from_documents(documents=chunked_documents, embedding=embeddings, persist_directory=persist_directory)

            # Additional code to process or store the results can be added here

        except Exception as e:
            print(f"An error occurred with chunk_size={chunk_size} and chunk_overlap={chunk_overlap}: {e}")




