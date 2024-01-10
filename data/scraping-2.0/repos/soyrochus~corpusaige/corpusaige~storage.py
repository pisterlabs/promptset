#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corpusaige is a Python tool (and utility library) enabling AI-powered systems analysis 
through deep exploration and understanding of comprehensive document sets and source code.
@copyright: Copyright © 2023 Iwan van der Kleijn
@license: MIT
"""
from pathlib import Path
from typing import List, Protocol
from corpusaige.config.read import CorpusConfig
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from corpusaige.documentset import Document, DocumentSet, Entry, FileType
from corpusaige.exceptions import InvalidParameters
from corpusaige.providers import vectorstore_factory


class Repository(Protocol):
    def add_docset(self, docset: DocumentSet):
        ...
    def remove_docset(self, docset_name: str):
        ...
    def add_doc(self, doc: Document, docset_name: str = ""):
        ...
    def search(self, search_str: str, results_num:int) -> List[str]:
        ...
    def ls(self, all_docs: bool = False, doc_set:str = '') -> List[str]:
        ...
        
class VectorRepository(Repository):
    def __init__(self, config: CorpusConfig):
        self.config = config
        self.vectorstore = vectorstore_factory(config)
    
    def as_retriever(self):
        return self.vectorstore.as_retriever()
    
    def get_glob(self, entry: Entry) -> str:
        if entry.recursive:
            return f"./**/*.{entry.file_extension}"
        else:
            return f"./*.{entry.file_extension}"    
    
    def add_doc(self, doc: Document, doc_set_name: str = ""):
        if doc.file_type == FileType.TEXT:
                
            text_splitter = RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=200)
            # Load text data from a file using TextLoader
            loader = TextLoader(str(doc.path))
            res = loader.load()
            document = None
            if len(res) > 0:
                document = res[0]
                document.metadata['doc-set'] = doc_set_name
            else:
                raise InvalidParameters(f"Could not load document from {doc.path}")
            
            chunks = text_splitter.split_documents([document])
        else:
            raise NotImplementedError(f'File type {doc.file_type} not supported yet.')

        self.vectorstore.add_documents(chunks)
        self.vectorstore.persist()
    
    def add_docset(self, doc_set: DocumentSet):
       
        chunks = None
        for entry in doc_set.entries:
            if entry.file_type == FileType.TEXT:
                
                text_splitter = RecursiveCharacterTextSplitter (chunk_size=1000, chunk_overlap=200)
                loader = DirectoryLoader(str(entry.path), self.get_glob(entry))
                docs = loader.load()
                for doc in docs:
                    doc.metadata['doc-set'] = doc_set.name
                    doc.metadata['path'] = str(Path(doc.metadata['source']).relative_to(entry.path))
                if chunks is None:
                    chunks = text_splitter.split_documents(docs)
                else:
                    chunks.extend(text_splitter.split_documents(docs))
            else:
                raise NotImplementedError(f'File type {entry.file_type} not supported yet.')
        self.vectorstore.add_documents(chunks)
        self.vectorstore.persist()

    def remove_docset(self, docset_name: str):
        #verify that docset exists
        if len(self.vectorstore.get(where={'doc-set': docset_name})['ids']) > 0:
            #cannot use vectorstore.delete(), have to resort to direct access to the collection
            self.vectorstore._collection.delete(where={'doc-set': docset_name})
        else:
            raise InvalidParameters(f"Could not find document set '{docset_name}'")
        
    def search(self, search_str: str, results_num: int) -> List[str]:
        result = self.vectorstore.similarity_search(search_str, k=results_num)
        #return [doc.page_content for doc in result]
        return ["\n\n".join([doc.metadata['source'],doc.page_content]) for doc in result]
    
    def ls(self, all_docs: bool = False, doc_set:str = '') -> List[str]:
        result = None
        if all_docs:
            result = self.vectorstore.get()
            sources = [metadata['path'] for metadata in result['metadatas']]
            #sources = [metadata['source'] for metadata in result['metadatas']]
            #remove duplicates from list
            return list(dict.fromkeys(sources))
        elif not all_docs and not doc_set:
            result = self.vectorstore.get()
            doc_sets = [metadata['doc-set'] for metadata in result['metadatas']]
            #remove duplicates from list
            return list(dict.fromkeys(doc_sets))
        else: 
            result = self.vectorstore.get(where={'doc-set': doc_set})
            sources = [metadata['path'] for metadata in result['metadatas']]
            #sources = [metadata['source'] for metadata in result['metadatas']]
            #remove duplicates from list
            return list(dict.fromkeys(sources))
            
        
        