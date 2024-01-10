import pandas as pd
import os
import requests
import sys
import time
from datasets import load_dataset, DatasetDict
from dotenv import load_dotenv
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, UnstructuredURLLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import CTransformers
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant


# Load OpenAI access and other custom paths
sys.path.append(os.path.abspath('../../src'))
from azure_openai_conn import OpenAIembeddings, qdrant_load_by_chunks

# Load environment variables
load_dotenv()

# Alternative embedding
# from fastembed.embedding import FlagEmbedding as Embedding
# embedding_model = Embedding(model_name="BAAI/bge-base-en", max_length=512) 
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


pdf_folder_path = destination_folder
documents = []
for file in os.listdir(pdf_folder_path)[:5]:
    print(file)
    if file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder_path, file)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

'''
ULTABEAUTY_2023Q4_EARNINGS.pdf
COCACOLA_2022_10K.pdf
GENERALMILLS_2022_10K.pdf
JPMORGAN_2022_10K.pdf
AMCOR_2022_8K_dated-2022-07-01.pdf
'''

# Load Embeddings: Many Problems for Exceed call rate
embeddings = OpenAIembeddings()
# Spliter
# todo: smarter spliter
# https://github.com/Azure-Samples/azure-search-openai-demo/blob/main/scripts/prepdocslib/textsplitter.py
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100, add_start_index=True)
# Generate Splits
chunked_documents = text_splitter.split_documents(documents)
# Chunk size and overlap
chunk_size=1500
overlap=100
# Initialize Qdrant database
qdrant = Qdrant.from_documents(documents=chunked_documents, embedding=embeddings, path='db_qdrant', collection_name="financebench")

query = "What is the Coca Cola Balance Sheet?"
docs = qdrant.similarity_search(query)
print(docs[0].page_content)

# https://github.com/langchain-ai/langchain/issues/11471
'''
collection_name = "financebench"  # replace with your collection name
qdrant_2 = Qdrant.construct_instance(
    texts=[],  # no texts to add
    embedding=embeddings,
    collection_name=collection_name,
)
'''