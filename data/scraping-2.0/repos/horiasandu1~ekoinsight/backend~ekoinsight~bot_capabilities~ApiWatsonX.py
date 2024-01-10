import os
from .ApiIbm import ApiIbm
from ..templates import *
from langchain import PromptTemplate
from langchain.llms import OpenAI

from ibm_watson_machine_learning.foundation_models import Model
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from langchain.chains import RetrievalQA

class ApiWatsonX(ApiIbm):
    def __init__(self,dry_run=False):
        super().__init__(dry_run=dry_run)
        self.provider_name="WatsonX"
        self.dry_run_object=dry_run_story
        self.as_string=True
        self.language="English"

    def set_language(self,language):
        self.language=language

    def fetch_prompt(self,item="metal can",prompt_template='pollution_prompt_template',max_tokens=40,stop_sequences=['END']):
        if not self.dry_run:
            if not prompt_template.startswith=="ibm_":
                prompt_template="ibm_"+prompt_template
        
            prompt_template=eval(prompt_template)
        
            prompt_txt = prompt_template+f" {item} | {self.language}"
            gen_parms_override = {"max_new_tokens": max_tokens,"stop_sequences":stop_sequences}
            generated_response = self.model.generate( prompt_txt, gen_parms_override)
            generated_response=[response['generated_text'] for response in generated_response['results']]
            return generated_response[0]

        else:
            return eval("dry_run_"+prompt_template)

    def translate(self,word,language=None,prompt_template='translate',max_tokens=40,stop_sequences=["END"]):

        if not language:
            language=self.language

        if not self.dry_run:
            if not prompt_template.startswith=="ibm_":
                prompt_template="ibm_"+prompt_template
        
            prompt_template=eval(prompt_template)
        
            prompt_txt = prompt_template+f" {word} | {language}"
            gen_parms_override = {"max_new_tokens": max_tokens,"stop_sequences":stop_sequences}
            generated_response = self.model.generate( prompt_txt, gen_parms_override)
            generated_response=[response['generated_text'] for response in generated_response['results']]
            return generated_response

        else:
            return eval("dry_run_"+prompt_template)

    
    def fetch_using_index(self,query = "in 1990, how much paper was generated?",language=None):
        """
        call self.produce_index_pinecone(self,index_name='cfc',docs_dir="recycling_data_dir",embeddings=None) first
        """
        if not language:
            language=self.language
        
        if not self.dry_run:
            watson_llm = WatsonxLLM(model=self.model)

            self.produce_index_pinecone(index_name='cfc',docs_dir="recycling_data")

            qa = RetrievalQA.from_chain_type(llm=watson_llm, chain_type="stuff", return_source_documents=True ,retriever=self.index.as_retriever(search_type='mmr',search_kwargs={"k":1}))

            response = qa({"query": query}, return_only_outputs=True)

            if "I don't know" in response['result']:
                return {'result':None,'source':None}
            result=response['result']
            source=response['source_documents'][0].metadata['source']

            return {'result':result,'source':source}
        
        else:
            return dry_run_index_fetch



