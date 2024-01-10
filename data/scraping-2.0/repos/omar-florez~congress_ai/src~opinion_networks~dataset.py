"""Node that run map-reduce on opinions"""
from langchain import OpenAI, PromptTemplate, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.mapreduce import MapReduceChain
from langchain.prompts import PromptTemplate

from langchain.chains import ReduceDocumentsChain
from langchain.text_splitter import RecursiveCharacterTextSplitter

import os
import pdb
from langchain.chains.combine_documents.stuff import StuffDocumentsChain

import sys
import os
sys.path.append(".")
# from opinion_networks.combine_documents.map_reduce import MapReduceDocumentsChain2
from langchain.chains import MapReduceDocumentsChain
from opinion_networks.prompts import law_prompt
from opinion_networks.opinion import OpinionPair

from typing import Optional, Any, Dict, List
import json

class Summary:
    def __init__(self, output_folder, llm=None):
        self.output_folder = output_folder
        """max_tokens: The maximum number of tokens to generate in the completion.
        -1 returns as many tokens as possible given the prompt and
        the models maximal context size."""
        if llm is None:
            self.llm = OpenAI(temperature=0.0, model_name="text-davinci-003", max_tokens=1000)
        else:
            self.llm = llm

    def __call__(self, file_path: str, language: str, overwrite=False) -> OpinionPair:
        file_name = os.path.basename(file_path)
        output_file = os.path.join(self.output_folder, file_name)
        if os.path.exists(output_file) and not overwrite:
            print(f'Load existing summary: {output_file}')
            with open(output_file, 'r') as f:
                summary = f.read()
            return summary

        print(f'Write new summary: {output_file}')
        with open(file_path, 'r') as f:
            raw_text = f.read()           
        
        # LLM to use in map and reduce stages         
        map_llm_chain = LLMChain(llm=self.llm, prompt=law_prompt.MAP_PROMPT)
        reduce_llm_chain = LLMChain(llm=self.llm, prompt=law_prompt.REDUCE_PROMPT)

        # Takes a list of documents and combines them into a single string
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=reduce_llm_chain,
            document_variable_name="text",
        )

        # Combines and iteravely reduces the mapped documents 
        reduce_documents_chain = ReduceDocumentsChain(
                # This is final chain that is called.
                combine_documents_chain=combine_documents_chain,
                # If documents exceed context for `combine_documents_chain`
                collapse_documents_chain=combine_documents_chain,
                # The maximum number of tokens to group documents into
                token_max=3000)

        # Combining documents by mapping a chain over them, then combining results with reduce chain
        combine_documents = MapReduceDocumentsChain(
            # Map chain
            llm_chain=map_llm_chain,
            # Reduce chain
            reduce_documents_chain=reduce_documents_chain,
            # The variable name in the llm_chain to put the documents in
            document_variable_name="text",
            #return_intermediate_steps=True
        )

        map_reduce = MapReduceChain(
            combine_documents_chain=combine_documents,
            text_splitter=RecursiveCharacterTextSplitter(    
                separators=["\n\n", "\n"], 
                chunk_size=5000, 
                chunk_overlap=350
            )
        )
        summary = map_reduce.run(input_text=raw_text, language=language)
        
        output_path = os.path.join(self.output_folder, file_name)
        with open(output_path, 'w') as f:
            f.write(summary)
        return summary

class LawDataset:
    def __init__(self, raw_text_root, crawled_files_root, summaries_root, llm=None):
        self.raw_text_root = raw_text_root
        self.crawled_files_root = crawled_files_root
        self.summaries_root = summaries_root
        self.llm = llm

    def get_law_text_paths(self, raw_text_root: Optional[str] = None) -> List[str]:
        if raw_text_root is None:
            files = [
                './data/peru/laws/texts/00336.txt',
                './data/peru/laws/texts/00350.txt',
                './data/peru/laws/texts/00349.txt',
                './data/peru/laws/texts/00180.txt',
            ]
            return files
        
        files = [os.path.join(raw_text_root, x) for x in os.listdir(raw_text_root) if x.endswith('.txt')]
        return files

    def get_law_labels(self, crawled_files_root):
        """Parse (law code, label) dictionary entries, 0: rejected, 1: approved."""

        law_labels = {}
        file_paths = [x for x in os.listdir(crawled_files_root) if x.endswith('.jsonl')]
        for file_name in file_paths:
            with open(os.path.join(crawled_files_root, file_name)) as f:
                for line in f.readlines():
                    law_info = json.loads(line)
                    key = law_info['codigo_ley']
                    value = law_info['estado_ley']
                    if value.lower() == 'al archivo':
                        law_labels[key] = 0
                    if value.lower() == 'publicado el peruano':
                        law_labels[key] = 1

        """
        labeled laws = 7195/8104 (88.78%)
        Counter([law_labels[x] for x in law_labels])
        Counter({
            'Al Archivo': 5030, 
            'Publicado El Peruano': 2165, 
            'Presentado': 362, 
            'Retirado': 155, 
            'En comisión': 147, 
            'Dictamen Negativo': 66, 
            'Orden del Día': 61, 
            'Dictamen': 55, 
            'Autógrafa': 13, 
            'Observado': 11, 
            'Rechazado de Plano': 10, 
            'Aprobado': 8, 
            'En comisiˇn': 4, 
            'Dispensado 2da. votación': 3, 
            'Se inhibe': 3, 
            'En comisiµn': 2, 
            'Aprobado en Primera Votación': 1, 
            'Devuelto': 1, 'En comisi¾n': 1, 
            'Orden del DÝa': 1, 
            'Orden del DÌa': 1, 
            'Anulado': 1, 
            'Orden del DŪa': 1, 
            'Reconsideración': 1, 
            'En comisiůn': 1
        })
        """
        return law_labels
    
    def load(self):
        summary = Summary(self.summaries_root, llm=self.llm)
        files = self.get_law_text_paths(raw_text_root=self.raw_text_root)
        law_labels = self.get_law_labels(crawled_files_root=self.crawled_files_root)
        
        x = [summary(file_path, language='Spanish', overwrite=False) for file_path in files]
        y = [law_labels[os.path.splitext(os.path.basename(x))[0]] for x in files] 
        return x, y
        

if __name__ == "__main__":
    import os
    os.environ["SERPAPI_API_KEY"] = ""
    os.environ["OPENAI_API_KEY"] = ""
    openai_api_key = ""

    raw_text = open('./data/peru/laws/pdfs/00336.txt').read()
    language = "Spanish"
    summary = Summary()
    output = summary(raw_text, language)
    print(output)
    