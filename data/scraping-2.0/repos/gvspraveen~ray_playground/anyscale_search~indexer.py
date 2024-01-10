from anyscale_docs_crawler import scrape_urls
from models import (
    hf_embed_model, 
    persist_dir
)
from langchain.document_loaders import WebBaseLoader
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS

import ray
import numpy as np
import time

from typing import Dict

# Inspired by https://www.anyscale.com/blog/build-and-scale-a-powerful-query-engine-with-llamaindex-ray
# https://gist.github.com/amogkam/8d2f10c8f6e2cba96673ada6c69311a9

# Step 1: Logic for parsing the web pages into llama_index documents.
def parse_urls(url_row: Dict[str, str]) -> Dict[str, Document]:
    url = url_row["path"]
    loader = WebBaseLoader(url)
    data = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=8000, separator="\n")
    all_splits = text_splitter.split_documents(data)
    return [{"doc": split} for split in all_splits]


if __name__ == "__main__":

    # Processing Logic Begins here
        
    urls = list(scrape_urls())
    # documents = SimpleWebPageReader(html_to_text=True).load_data(urls[0:1])
    # print(documents[0])

    all_urls = [{"path": url} for url in urls]

    # Create the Ray Dataset pipeline
    ds = ray.data.from_items(all_urls)

    # Parallel process the urls and parse webpage and create Documents
    loaded_ds = ds.flat_map(parse_urls)

    documents = []

    for row in loaded_ds.iter_rows():
        documents.append(row["doc"])
    print("Length of rows: ", len(documents))

    @ray.remote(num_gpus=1)
    def process_shard(shard): 
        result = FAISS.from_documents(shard, hf_embed_model)
        return result

    def process_docs(db_shards = 8):
        print(f'Loading chunks into vector store ... using {db_shards} shards') 
        st = time.time()
        shards = np.array_split(documents, db_shards)
        futures = [process_shard.remote(shards[i]) for i in range(db_shards)]
        results = ray.get(futures)
        et = time.time() - st
        print(f'Shard processing complete. Time taken: {et} seconds.')


        st = time.time()
        print('Merging shards ...')
        # Straight serial merge of others into results[0]
        db = results[0]
        for i in range(1,db_shards):
            db.merge_from(results[i])
        et = time.time() - st
        print(f'Merged in {et} seconds.') 
        db.save_local(persist_dir)
    process_docs()

    print("Vector index successfully Saved. Ready for serving.")
