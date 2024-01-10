
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader

from elasticsearch import Elasticsearch

def parse_book(filepath):
    loader = TextLoader(filepath)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
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

    with Elasticsearch([url], verify_certs=True) as es:
        ## Parse the book if necessary
        if not es.indices.exists(index=index_name):
            print(f'\tThe index: {index_name} does not exist')
            print(">> 1. Chunk up the Source document")
            
            results = parse_triplets(filepath)

            print(">> 2. Index the chunks into Elasticsearch")
            
            db.from_texts(results, embedding=hf, elasticsearch_url=url, index_name=index_name)
        else:
            print("\tLooks like the book is already loaded, let's move on")

def loadBookBig(filepath, url, hf, db, index_name):

    with Elasticsearch([url], verify_certs=True) as es:
        ## Parse the book if necessary
        if not es.indices.exists(index=index_name):
            print(f'\tThe index: {index_name} does not exist')
            print(">> 1. Chunk up the Source document")
            
            docs = parse_book(filepath)

            print(">> 2. Index the chunks into Elasticsearch")
            
            # list_of_strings = [str(d) for d in docs]

            db.from_documents(docs, embedding=hf, elasticsearch_url=url, index_name=index_name)
        else:
            print("\tLooks like the book is already loaded, let's move on")

