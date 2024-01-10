
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

## for vector store
from langchain.vectorstores import ElasticVectorSearch
from elasticsearch import Elasticsearch
from config import Config

with open('simple.cfg') as f:
    cfg = Config(f)

fingerprint = cfg['ES_FINGERPRINT']
endpoint = cfg['ES_SERVER']
username = "elastic"
password = cfg['ES_PASSWORD']
ssl_verify = {
    "verify_certs": True,
    "basic_auth": (username, password),
    "ca_certs": "./python_es_client.pem",
}

url = f"https://{username}:{password}@{endpoint}:9200"

def parse_book(filepath):
    loader = TextLoader(filepath)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=384, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    return docs

def parse_triplets(filepath):
    docs = parse_book(filepath)
    result = []
    for i in range(len(docs) - 2):
        concat_str = docs[i].page_content + " " + docs[i+1].page_content + " " + docs[i+2].page_content
        result.append(concat_str)
    return result
    #db.from_texts(docs, embedding=hf, elasticsearch_url=url, index_name=index_name)

## load book utility
## params
##  filepath: where to get the book txt ... should be utf-8
##  url: the full Elasticsearch url with username password and port embedded
##  hf: hugging face transformer for sentences
##  db: the VectorStore Langcahin object ready to go with embedding thing already set up
##  index_name: name of index to use in ES
##
##  will check if the index_name exists already in ES url before attempting split and load
def loadBookTriplets(filepath, url, hf, db, index_name):
    with open('simple.cfg') as f:
        cfg = Config(f)
    
    fingerprint = cfg['ES_FINGERPRINT']
    es = Elasticsearch( [ url ], 
                    basic_auth = ("elastic", cfg['ES_PASSWORD']), 
                    ssl_assert_fingerprint = fingerprint, 
                    http_compress = True  )
   
    ## Parse the book if necessary
    if not es.indices.exists(index=index_name):
        print(f'\tThe index: {index_name} does not exist')
        print(">> 1. Chunk up the Source document")
        
        results = parse_triplets(filepath)

        print(">> 2. Index the chunks into Elasticsearch")
        
        elastic_vector_search= ElasticVectorSearch.from_documents( docs,
                                embedding = hf, 
                                elasticsearch_url = url, 
                                index_name = index_name, 
                                ssl_verify = ssl_verify)
    else:
        print("\tLooks like the book is already loaded, let's move on")

def loadBookBig(filepath, url, hf, db, index_name):    
    
    es = Elasticsearch( [ url ], 
                       basic_auth = ("elastic", cfg['ES_PASSWORD']), 
                       ssl_assert_fingerprint = fingerprint, 
                       http_compress = True  )
    
    ## Parse the book if necessary
    if not es.indices.exists(index=index_name):
        print(f'\tThe index: {index_name} does not exist')
        print(">> 1. Chunk up the Source document")
        
        docs = parse_book(filepath)
        
        # print(docs)

        print(">> 2. Index the chunks into Elasticsearch")
        
        elastic_vector_search= ElasticVectorSearch.from_documents( docs,
                                embedding = hf, 
                                elasticsearch_url = url, 
                                index_name = index_name, 
                                ssl_verify = ssl_verify)   
    else:
        print("\tLooks like the book is already loaded, let's move on")

