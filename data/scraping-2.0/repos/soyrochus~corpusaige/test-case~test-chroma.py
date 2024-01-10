#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corpusaige is a Python tool (and utility library) enabling AI-powered systems analysis 
through deep exploration and understanding of comprehensive document sets and source code.
@copyright: Copyright © 2023 Iwan van der Kleijn
@license: MIT
"""

# Import necessary modules

from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
import os
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.document_loaders import DirectoryLoader

from corpusaige.config.read import get_config
from corpusaige.corpus import CorpusData, CorpusReader
from corpusaige.documentset import DocumentSet

path = Path(__file__).parent.absolute()

def test_chroma():
    
    docs_loader = DirectoryLoader('./test-case/single-file/', glob="./*.md")
    docs = docs_loader.load()
    print(f'files: {len(docs)} ')

    text_splitter = RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=200)

    chunks = text_splitter.split_documents(docs)
    print(f'chunks: {len(chunks)} ')
    
    persist_directory = './test-case/test-chroma.db'
    embedding = OpenAIEmbeddings(openai_api_key=os.environ['OPENAI_API_KEY']) 
    single_chunk = [chunks.pop(0)]
    vectordb = Chroma.from_documents(documents=single_chunk,
                                    embedding=embedding,
                                    persist_directory=persist_directory)
    print(f"There are {vectordb._collection.count()} docs in the collection")
    vectordb.persist()
    
    db = Chroma(embedding_function=embedding, persist_directory=persist_directory)
    print(f"There are {db._collection.count()} docs in the collection")
    
    ids = db.add_documents(chunks)
    db.persist()
    
    print(f'ids: {ids}')
    print(f"There are {db._collection.count()} docs in the collection")
    
def test_write_corpusaige():
    _test_path = path / 'test-single-file'
    config = get_config(_test_path)
    name="test-corpus"
    doc_paths = [_test_path]
    doc_types = ["text:md"]
    docset = DocumentSet.initialize(name, doc_paths, doc_types, False)
    corpusdata = CorpusData(config)
    corpusdata.add_docset(docset)
    
def test_read_corpusaige():
    _test_path = path / 'test-single-file'
    config = get_config(_test_path)
    corpus = CorpusReader(config)
    corpus.send_prompt("What is the prelude?", show_sources=True, print_output=True)

if __name__ == '__main__':
    #test_chroma()
    test_write_corpusaige()
    #test_read_corpusaige()
    
