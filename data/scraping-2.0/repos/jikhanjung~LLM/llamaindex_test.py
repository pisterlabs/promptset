import os.path
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    ServiceContext,
)
from ZWrapper import ZWrapper, gDatabase, CollectionCache, ItemCache, LastVersion, CollectionItemRel
import json, os
from pyzotero import zotero
from openai import OpenAI
from dotenv import load_dotenv # pip install python-dotenv
load_dotenv()

import logging
import sys

#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
#logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

zotero_api_key = os.environ.get("ZOTERO_API_KEY")
zotero_user_id = os.environ.get("ZOTERO_USER_ID")

zot = zotero.Zotero(zotero_user_id, 'user', zotero_api_key)

def load_data_from_dir(directory):
    if not os.path.exists("./storage"):
        #llm = OpenAI(model="gpt-4")
        service_context = ServiceContext.from_defaults()
        documents = SimpleDirectoryReader(directory).load_data()
        index = VectorStoreIndex.from_documents(documents, service_context=service_context)
        index.storage_context.persist()
    else:
        # load the existing index
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
    #query_engine = index.as_query_engine()
    query_engine = index.as_query_engine(similarity_top_k=5)
    return query_engine


def prepare_data_directory(collection_key, data_directory):
    colcache = CollectionCache.get_or_create(key=collection_key)[0]
    cirel_list = CollectionItemRel.select().where(CollectionItemRel.collection == colcache)
    for cirel in cirel_list:
        itemcache = ItemCache.get_or_create(key=cirel.item.key)[0]
        print(itemcache.key)
        # json to dict
        item_data = json.loads(itemcache.data)
        if 'contentType' in item_data and item_data['contentType'] == 'application/pdf':
            # download pdf and dump file to data_directory
            filepath = data_directory + '/' + item_data['filename']
            if not os.path.exists(filepath):
                zot.dump(item_data['key'],item_data['filename'],data_directory)
            #print(item['data'])


if __name__ == '__main__':
    # get collection key input from user
    collection_key = input("Enter collection key: ")
    data_directory = './pdfs/' + collection_key
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    prepare_data_directory(collection_key, data_directory)
    query_engine = load_data_from_dir(data_directory)
    while True:
        query = input("Enter query: ")
        if query == 'exit':
            break
        response = query_engine.query(query)
        print(response)
