from pathlib import Path
import yaml
import os
import pandas as pd
import sys
# from legacy_code_assistant.knowledge_base.description_generator import CodeConditionedGenerator
from legacy_code_assistant.knowledge_base.knowledge_builder import KnowledgeBaseBuilder
# from legacy_code_assistant.knowledge_base.knowledge_builder import CodeAnalyzer
from langchain.embeddings import AzureOpenAIEmbeddings

from prompts import modifyPrompt, analyzePrompt, addPrompt, testPrompt, vulnerabilityPrompt

from langchain.chat_models import AzureChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.vectorstores import FAISS

with open('notebooks/credentials.yaml', "r") as f:
    credentials = yaml.load(f, Loader=yaml.FullLoader)
os.environ["AZURE_OPENAI_ENDPOINT"] = credentials['AZURE_OPENAI_ENDPOINT']
os.environ["AZURE_OPENAI_API_KEY"] = credentials['AZURE_OPENAI_API_KEY']


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


class pipeProcess:
    def __init__(self, filepath, index_name):
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


    def _build_chain(self, template):
        prompt = ChatPromptTemplate.from_template(template)

        chain = (
            {"context": self.retriever | format_docs,
                "question": RunnablePassthrough()}
            | prompt
            | self.model
            | StrOutputParser()
        )
        return chain

    def analyzePipe(self):
        print("Wchodzę w pipe analizy, Wpisz swój prompt do chatu")
        user_input = input()

        # print("znalazlem",self.kbb_docs.search(user_input),)
        template = analyzePrompt

        chain = self._build_chain(template)
        result = chain.invoke(user_input)
        print(result)
        return result

    def addPipe(self):
        print("Wchodzę w pipe dodawania, Wpisz swój prompt do chatu")
        user_input = input()

        # print(self.kbb_docs.search(user_input))
        template = addPrompt

        chain = self._build_chain(template)
        result = chain.invoke(user_input)
        print(result)
        return result

    def modifyPipe(self):
        print("Wchodzę w pipe modyfikacji, Wpisz swój prompt do chatu")
        user_input = input()
        # print("retriever",self.retriever,"||",self.kbb_docs.search(user_input))
        template = modifyPrompt
        chain = self._build_chain(template)
        result = chain.invoke(user_input)
        print(result)
        return result

    def testPipe(self):
        print("Wchodzę w pipe pisanie testów, Wpisz swój prompt do chatu")
        user_input = input()
        template = testPrompt
        chain = self._build_chain(template)
        result = chain.invoke(user_input)
        print(result)
        return result

    def vulPipe(self):
        print("Wchodzę w pipe sprawdzanie podatności, Wpisz swój prompt do chatu")
        user_input = input()
        template = vulnerabilityPrompt
        chain = self._build_chain(template)
        result = chain.invoke(user_input)
        print(result)
        return result

    def otherPipe(self):
        print("Wchodzę w pipe inne, Wpisz swój prompt do chatu")
        user_input = input()
        template = """Answer the question based only on the following context:
        {context}

        Question: {question}
        """
        chain = self._build_chain(template)
        result = chain.invoke(user_input)
        print(result)
        return result

    def startPipe(self):
        print("Wybierz jedną z mozliwych kategorii: 1: Analiza, 2: Dodanie, 3: Testy: 4, Modyfikacja,5: Szukanie podatności 6: Inne. Wpisz numer kategorii: ")
        category = input()
        print("Wybrana kategoria to " + category)
        if (category == "1"):
            self.analyzePipe()
        elif (category == "2"):
            self.addPipe()
        elif (category == "3"):
            self.modifyPipe()
        elif (category == "4"):
            self.testPipe()
        elif (category == "5"):
            self.vulPipe()
        elif (category == "6"):
            self.otherPipe()


if __name__ == "__main__":
    pipe = pipeProcess(filepath='notebooks/generated_docstrings.csv',
                       index_name='notebooks/docstring_based_index')
    pipe.startPipe()
