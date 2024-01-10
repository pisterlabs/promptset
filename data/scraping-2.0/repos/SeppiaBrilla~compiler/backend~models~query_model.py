from models.model import LLM_model
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain.docstore.document import Document
from langchain.chains.question_answering import load_qa_chain
import logging

class Query_model(LLM_model):
    def __init__(self, model:str) -> None:
        generator = pipeline("text-generation", model=model, max_new_tokens=50, device_map="auto")
        generator.model.config.pad_token_id = generator.model.config.eos_token_id
        llm = HuggingFacePipeline(pipeline=generator)
        self.documents = []
        self.chain = load_qa_chain(llm=llm)
    
    def query(self, query:str) -> str:
        query = self.__get_query(query)
        response = self.chain.run(input_documents=self.documents, question=query)
        return response.replace('[[:alnum:]]([\\s]{2,})', '')

    def add_documents(self, *documents:Document) -> None:
        self.documents += documents
    
    @staticmethod
    def __get_query(query:str) -> str:
        return f'''
        This data describes a set of command you can use to fullfill some requests. you can use them by calling the function followed by the charachter ":" and its parameters. 
        The function call must be sourranded by the charachters _$ and $_. If the call contains the value "(None)" it means there are no parameters. 
        Example:
            function = list-folder-content
            correct call = _$list-folder-content:folder$_

        {query}'''



def get_queryModel(model_name:str) -> Query_model:
    logging.info(f"initializating a query model with model: {model_name}")
    return Query_model(model_name)
