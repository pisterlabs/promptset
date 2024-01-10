# Class to generate queries to the GraphQL API using LangChain from natural language questions
import json
from webbrowser import Chrome
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.schema import BaseLanguageModel


class QueryConverter:
    def prompt_template(self):
        prompt_template = """
        Given an input question, Convert into a syntactically correct GraphQL Query. 
        Do not include `nodes` in the query.
        
        # GraphQL Reference

        ```
        {context}
        ```

        # Question
        {question}
        """

        return PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

    def __init__(self, docsearch: Chrome, llm: BaseLanguageModel):
        chain_type_kwargs = {"prompt": self.prompt_template()}
        self.qa = RetrievalQA.from_chain_type(
            llm=llm, chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 4}), chain_type_kwargs=chain_type_kwargs)

    def run(self, question):
        response = self.qa.run(question)
        # extract the contents within {} from the response string
        print(f"raw response: {response}")
        response = response[response.find("query {"): response.rfind("}") + 1]
        print(f"split response: {response}")

        return response
