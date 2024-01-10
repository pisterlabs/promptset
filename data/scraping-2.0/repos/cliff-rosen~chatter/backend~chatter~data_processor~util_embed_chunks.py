import openai
from openai.embeddings_utils import get_embedding, cosine_similarity
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import pinecone
import json
import re
import os
import sys
sys.path.append('db')
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import db
import local_secrets as secrets

"""
TO DO:
log error if upsert response not {'upserted_count': 1}
upsert in batches

"""
PINECONE_API_KEY = secrets.PINECONE_API_KEY
INDEX_NAME = "main-index"
OPENAI_API_KEY = secrets.OPENAI_API_KEY
openai.api_key = OPENAI_API_KEY

pinecone.init(api_key=PINECONE_API_KEY, environment="us-east1-gcp")
index = pinecone.Index(INDEX_NAME)

def get_openai_embedding(text):
    embedding_model = "text-embedding-ada-002"
    return get_embedding(
        text,
        engine="text-embedding-ada-002"
    )

def run():
    print("Starting embedding update for domain", domain_id)
    conn = db.get_connection()
    rows = db.get_document_chunks(conn, domain_id)
    cur_count = 1
    tot_count = len(rows)
    print("Total chunks to be updated", tot_count)
    for row in rows:
        doc_chunk_id = row['doc_chunk_id']
        chunk_text = row['chunk_text']
        embedding = get_openai_embedding(chunk_text)
        print("Processing ", cur_count, " of ", tot_count)
        print("  Data: ", doc_chunk_id, embedding[:10])
        db.update_document_chunk_embedding(conn, doc_chunk_id, embedding)
        cur_count = cur_count + 1
    db.close_connection(conn)

def fetch():
    res = index.fetch(ids=['3'])
    print(res['vectors']['3']['metadata'])

# runtime settings
domain_id = 27

print(index.describe_index_stats())
run()
