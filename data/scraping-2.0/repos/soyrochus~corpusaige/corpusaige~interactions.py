#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Corpusaige is a Python tool (and utility library) enabling AI-powered systems analysis 
through deep exploration and understanding of comprehensive document sets and source code.
@copyright: Copyright © 2023 Iwan van der Kleijn
@license: MIT
"""

# Import necessary modules

from typing import Protocol
from corpusaige.config.read import CorpusConfig
from corpusaige.providers import llm_factory, vectorstore_factory
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA, ConversationalRetrievalChain


class Interaction(Protocol):

    def send_prompt(self, prompt: str) -> str:
        ...

# Cite sources

def process_llm_response(llm_response):
    print(llm_response['result'])
    print('\n\nSources:')
    for source in llm_response["source_documents"]:
        print(source.metadata['source'])
        
class StatelessInteraction(Interaction):
    def __init__(self, config: CorpusConfig):
        # create the chain to answer questions
        self.llm = llm_factory(config)
        self.vectorstore = vectorstore_factory(config)
        retriever = self.vectorstore.as_retriever()
        
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm,
                                                    chain_type="stuff",
                                                    retriever=retriever,
                                                    return_source_documents=True,
                                                    verbose=True)

    def send_prompt(self, prompt: str) -> str:
        llm_response = self.qa_chain(prompt)
        return process_llm_response(llm_response)
    
class StatefullInteraction(Interaction):
    def __init__(self, config: CorpusConfig, retriever = None):
        # create the chain to answer questions
        self.llm = llm_factory(config)
        if retriever is None:
            self.retriever = vectorstore_factory(config).as_retriever()
        else:
            self.retriever = retriever 
        
        self.memory = ConversationBufferMemory(memory_key="chat_history", 
                                               input_key='question', 
                                               output_key='answer', 
                                               return_messages=True)
        
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm, 
            retriever = self.retriever, 
            memory=self.memory, 
            return_source_documents=True)

    def send_prompt(self, prompt: str, show_sources: bool = False, results_num: int=4) -> str:
        
        self.retriever.search_kwargs['k'] = results_num
        llm_response = self.qa_chain({"question": prompt})
      
        if show_sources:
            sources = [f"doc-set: {source.metadata['doc-set']}, source: {source.metadata['source']}"  for source in llm_response["source_documents"]]
            sources_str = "\n | ".join(sources)
            return f'{llm_response["answer"]}\n\nSources: {sources_str}'
        else:
            return llm_response['answer']
     