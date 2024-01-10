#!/usr/bin/env python3
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.document_loaders import (
    TextLoader,
)
from python_api.db.milvus import delete_by_document_name, insert
from python_api.shared.app_base import initialize_openai
from langchain.embeddings import OpenAIEmbeddings

embeddings_model = 'text-embedding-ada-002'
chunk_size = 500
chunk_overlap = 50
embedding_chunk_size = 16

file_path = os.path.join('source_documents', "state_of_the_union.txt") 

def load_document() -> Document:
    loader_args = {"encoding": "utf8"}
    loader = TextLoader(file_path, **loader_args)
    return loader.load()
    
def chunk_document(document):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    texts = text_splitter.split_documents(document)
    return texts

def process_document() -> Document:
    document = load_document()
    return chunk_document(document)

def ingest():    
    initialize_openai()

    delete_by_document_name("source_documents/state_of_the_union.txt")
    texts = process_document()
    texts = texts[0:5]

    texts_content = [ document.page_content for document in texts ]
    output_dict = [ {
        "text": document.page_content,
        "source": document.metadata["source"],
    } for document in texts ]

    embeddings = []
    total_count = len(texts_content)
    for i in range(0, total_count, embedding_chunk_size):
        end = min(i + embedding_chunk_size, total_count)
        current_items = texts_content[i:end]
        embeddingsModel = OpenAIEmbeddings(
            model='text-embedding-ada-002')

        embeddings_data = embeddingsModel.embed_documents(current_items)
        tokens_array = [item for item in embeddings_data]
        embeddings = [*embeddings, *tokens_array]
        
    for i in range(total_count):
        output_dict[i]["vector"] = embeddings[i]

    pks = insert(output_dict)
    print(pks)
    print(f"Ingestion complete!")



