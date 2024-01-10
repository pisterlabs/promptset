
import lib_book_parse
import lib_llm
import lib_embeddings
import lib_vectordb

from lib_es_bulk import bulkLoadIndexPipeline

import os

from pathlib import Path
import pickle

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

from elasticsearch import Elasticsearch

from tqdm import tqdm


config = {
    "bookName" : "Wookieepedia",
    "bookIndexName": "book_wookieepedia_mpnet",
    "bookFilePath": "starwars_all_canon_data_*.pickle"
}

bookName = config['bookName']
bookFilePath = config['bookFilePath']
index_name = config['bookIndexName']

endpoint = os.getenv('ES_SERVER', 'ERROR') 
username = os.getenv('ES_USERNAME', 'ERROR') 
password = os.getenv('ES_PASSWORD', 'ERROR')

url = f"https://{username}:{password}@{endpoint}:443"


## Load the book

count = 0
with Elasticsearch([url], verify_certs=True) as es:
    files = sorted(Path('./Dataset').glob(bookFilePath))
    for fn in files:
        print(f"Starting book: {fn}")
        with open(fn,'rb') as f:
            part = pickle.load(f)
            batch = []
            for ix, (key, value) in tqdm(enumerate(part.items()), total=len(part)):
                title = value['title'].strip()
                sw_url = value['url']
                paragraphs = value['paragraph']
                for px, p in enumerate(paragraphs):
                    payload = {
                        "text": p,
                        "metadata":{
                            "title": title,
                            "url": sw_url,
                            "pg_num": px
                        }
                    }
                    # print(f"{ix} {px} {title}")
                    batch.append(payload)
                    count = count + 1
                if len(batch) >= 100:
                    bulkLoadIndexPipeline(batch,index_name,"sw-embeddings")
                    batch = []
            
            bulkLoadIndexPipeline(batch,index_name,"sw-embeddings")
            print(f"Count {count}")


