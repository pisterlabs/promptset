"""Node that run map-reduce on opinions"""
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate

from langchain.chains import ReduceDocumentsChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import pdb
import json
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

import sys
import os
from typing import List, Dict, Any, Optional
sys.path.append(".")
from opinion_networks.prompts import opinion_prompt, law_prompt
from langchain.docstore.document import Document
from opinion_networks.opinion import Opinion, OpinionPair

class Reasoner:
    def __init__(self, background: Optional = None, language: str='English', **kwargs):
        self.opinions = None
        if 'llm' not in kwargs or kwargs['llm'] is None:
            self.llm = OpenAI(temperature=kwargs['temperature'], model_name="text-davinci-003", max_tokens= 3000)
        else:
            self.llm = kwargs['llm']
        self.type = kwargs['type']
        self.background = background
        self.language = language

    def reduce_chain(self, opinions: list[OpinionPair]) -> OpinionPair:
        """Combines a list of opinions into a a list of of elements: a positive and negative opinion."""
        opinions_list = [
            Document(
                page_content=f"Score: {opinion.score}. {opinion.reasoning}", 
                metadata={'opinion_type': opinion.opinion_type}) 
            for out in opinions 
            for opinion in out.get_pos_neg_opinions()
        ]

        # `collapse_documents_chain` is used if the documents passed in are too many to all be passed to 
        # `combine_documents_chain` in one go. In this case. This method is called recursively on as big 
        # of groups of documents as are allowed.
        collapse_prompt = opinion_prompt.COLLAPSE_PROMPT
        collapse_llm_chain = LLMChain(llm=self.llm, prompt=collapse_prompt)
        collapse_documents_chain = StuffDocumentsChain(
            llm_chain=collapse_llm_chain, 
            document_variable_name="opinions"
        )

        # `combine_documents_chain`: This is final chain that is called
        # This chain takes a list of documents and first combines them into a single string.
        # It does this by formatting each document into a string with the `document_prompt`
        # and then joining them together with `document_separator`. It then adds that new
        # string to the inputs with the variable name set by `document_variable_name`.
        # Those inputs are then passed to the `llm_chain`.
        # this class does this in combine_docs(): 
        #       inputs = {}
        #       self.document_variable_name="opinions"
        #       inputs[self.document_variable_name] = self.document_separator.join(opinions_list)
        #       self.llm_chain.predict(callbacks=callbacks, **inputs), {}
        combine_prompt = opinion_prompt.REDUCE_PROMPT if self.type == 'hidden' else opinion_prompt.DECIDE_PROMPT
        reduce_llm_chain = LLMChain(llm=self.llm, prompt=combine_prompt)
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_llm_chain,
            document_variable_name="opinions"
        )

        #Internally combines and iteravely reduces the mapped documents 
        reduce_documents_chain = ReduceDocumentsChain(
                # This is final chain that is called.
                combine_documents_chain=combine_documents_chain,
                # If documents exceed context during `combine_documents_chain, recursively pack them into 
                # smaller summaries with `collapse_documents_chain until there is a summary that fits in context.
                # TODO: Test if a large collection of opinions will break the logic of collapsing opinions using reduce_prompt.COLLAPSE_PROMPT
                collapse_documents_chain=collapse_documents_chain,
                # The maximum number of tokens to group documents into
                token_max=3000)

        try:
            law_str = opinions[0].law
            output_json = reduce_documents_chain.run(
                input_documents=opinions_list, 
                law=law_str, 
                background=opinion_prompt.get_background(self.background), 
                language=self.language
            )   
            pdb.set_trace()
            opinions = OpinionPair(output_json, law_str)
        except: 
            pdb.set_trace()
        return opinions   

    """ Convert the summary of a law into Opinions, which is an objeact that contains a positve and negative opinion"""
    def read_chain(self, law_str) -> OpinionPair:
        prompt = opinion_prompt.READ_PROMPT.format(
            law=law_str, 
            background=opinion_prompt.get_background(self.background),
            language=self.language
        )
        
        try:
            output_json = self.llm(prompt)
            output_json = output_json[output_json.find('['): output_json.find(']')+1]
            opinions = OpinionPair(output_json, law_str)
            # extract_json_prompt = f"""Extract the array contained in this text:{output_json}\n\n\nOUTPUT:"""
            # output_json2 = self.llm(extract_json_prompt)
            # pp output_json2
            # Token indices sequence length is longer than the specified maximum sequence length for this model (1335 > 1024)
            # the current text generation call will exceed the model's predefined maximum length (4096)
        except: 
            pdb.set_trace()        
        return opinions      
    
    """ Convert `opinions` into Opinions, which is an objeact that contains a positve and negative opinion. 
    The incoming `opinions` can be law string or a list of `Opinions`, which is an object that contains a positve 
    and negative opinion."""
    def __call__(self, opinions: Any, background: Any) -> OpinionPair:
        if self.type == 'input':
            opinions = self.read_chain(law_str=opinions)
            print(f'>>> {self.type}@read_chain: {opinions}')
        else:
            opinions = self.reduce_chain(opinions=opinions)
            print(f'>>> {self.type}@reduce_chain: {opinions}')
        return opinions
        