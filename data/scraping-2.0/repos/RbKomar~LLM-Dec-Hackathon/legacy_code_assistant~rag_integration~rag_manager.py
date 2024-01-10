from pathlib import Path
import yaml
import os
import pandas as pd
import sys
# from legacy_code_assistant.knowledge_base.description_generator import CodeConditionedGenerator
from legacy_code_assistant.knowledge_base.knowledge_builder import KnowledgeBaseBuilder
# from legacy_code_assistant.knowledge_base.knowledge_builder import CodeAnalyzer
from langchain.embeddings import AzureOpenAIEmbeddings

from legacy_code_assistant.rag_integration.rag_prompts import (
    modifyPrompt, analyzePrompt, addPrompt, testPrompt, vulnerabilityPrompt)

from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores import FAISS


#   def getFunctionsDataFrame():
#       path = Path() / '..' / '..' / 'dziwne' / 'Django-School-Management-System'
#       paths = list(path.rglob('**/*.py'))
#       ca = CodeAnalyzer(paths)
#       results = ca.analyze()
#       df = pd.DataFrame(results)
#       print("pozyskalem baze danych")
#       return df

def format_docs(docs):
    return [doc.page_content for doc in docs]


class RagManager:
    def __init__(self, filepath, index_name, credentials_filepath):
        with open(credentials_filepath, "r") as f:
            credentials = yaml.load(f, Loader=yaml.FullLoader)
        os.environ["AZURE_OPENAI_ENDPOINT"] = credentials['AZURE_OPENAI_ENDPOINT']
        os.environ["AZURE_OPENAI_API_KEY"] = credentials['AZURE_OPENAI_API_KEY']


        self.model = AzureChatOpenAI(
            openai_api_version="2023-05-15",
            azure_deployment=credentials['Deployment_completion'],
        )
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=credentials['Deployment_embeddings'],
            openai_api_version="2023-05-15",
        )
        self.df = pd.read_csv(filepath)


        self.kbb_docs = KnowledgeBaseBuilder(index_name=index_name, model=self.embeddings)
        self.kbb_docs.load_index()
        self.retriever = self.kbb_docs.get_retriever()


    def _build_chain(self, template, question=None, context=None):
        prompt = ChatPromptTemplate.from_template(template)

        if context is None:
            chain = (
                {"context": self.retriever | format_docs,
                 "question": RunnablePassthrough()}
                | prompt
                | self.model
                | StrOutputParser()
            )
        else:
            chain = (
                {'context': RunnablePassthrough(), 
                 "question": RunnablePassthrough()}
                | prompt
                | self.model
                | StrOutputParser()
            )
        return chain
    
    def _run_chain(self, prompt_template, user_input, context=None):
        chain = self._build_chain(prompt_template, context=context)
        input_dict = {'question': user_input}
        if context is not None:
            input_dict['context'] = context
        result = chain.invoke(input_dict)
        return result

    def analyze_code(self, user_input, context=None):
        return self._run_chain(analyzePrompt, user_input, context=context)

    def add_code(self, user_input, context=None):
        return self._run_chain(addPrompt, user_input, context=context)

    def modify_code(self, user_input, context=None):
        return self._run_chain(modifyPrompt, user_input, context=context)

    def write_tests(self, user_input, context=None):
        return self._run_chain(testPrompt, user_input, context=context)

    def search_for_vulnerabilities(self, user_input, context=None):
        return self._run_chain(vulnerabilityPrompt, user_input, context=context)
    
    def refactor_code(self, user_input):
        raise NotImplementedError
    
