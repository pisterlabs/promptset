from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader


import os, json

with open('conf.json', 'r') as f:
    json_data = json.load(f)
    
os.environ['OPENAI_API_KEY'] = json_data['API_KEY']

from data.fetch_data import fetch_data_from_db
import pandas as pd

query = """
    SELECT pdf.etf_tkr, pdf.child_stk_tkr
        FROM os_pdf_info pdf
        INNER JOIN os_stk_info stk
        ON pdf.child_stk_tkr=stk.stk_tkr
"""

ticker_df = pd.DataFrame(fetch_data_from_db(query=query))

ticker_list = list(set(ticker_df['child_stk_tkr'].to_list()))

import yfinance as yf
from tqdm.auto import tqdm

yf_stk_info = yf.Tickers(" ".join(ticker_list))

for t in tqdm(ticker_list):

    if os.path.isfile(f"./stk_infos/{t}.json"):
        continue

    parent_etfs = ticker_df[ticker_df['child_stk_tkr']==t]['etf_tkr'].to_list()

    yf_stk_info.tickers[t].info['parent_etfs'] = parent_etfs

    with open(f'./stk_infos/{t}.json', 'w') as f:
        json.dump(yf_stk_info.tickers[t].info, f, indent=4)

from langchain.document_loaders import JSONLoader

def metadata_func(record: dict, metadata: dict) -> dict:
        
    for k, v in record.items():
        if k=='longBusinessSummary':
            continue
        
        if k == 'parent_etfs':
            metadata[k] = ",".join(record.get(k))
        
        if type(record.get(k)) not in [str, int, float]:
            continue
        
        metadata[k] = record.get(k)

    return metadata

loader = DirectoryLoader('./stk_infos',
                         glob='*.json', 
                         loader_cls=JSONLoader, 
                         loader_kwargs={'jq_schema': '.', 'content_key': 'longBusinessSummary', 'metadata_func': metadata_func})

documents = loader.load()

len(documents)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=200) #1000자 씩 끊되, 200자 씩 겹치게 만든다.
texts = text_splitter.split_documents(documents)

print(len(texts))

persist_directory='db'

embedding = OpenAIEmbeddings(
    model='text-embedding-ada-002'
)

vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embedding,
    persist_directory=persist_directory,
)
