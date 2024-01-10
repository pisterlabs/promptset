import os
from configparser import ConfigParser

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate

config = ConfigParser()
config.read("config.ini")

OPENAI_SECRET_KEY = config["openai"]["secret_key"]
MODEL = config["openai"]["model"]
TEMPERATURE = float(config["openai"]["temperature"])

os.environ["OPENAI_API_KEY"] = OPENAI_SECRET_KEY


class ReverseChainBaseClass:
    def __init__(self, model: str, temperature: float) -> None:
        self.model = model
        self.temperature = temperature
        self.llm = OpenAI(
            model_name=MODEL,
            temperature=self.temperature,
        )
        self.template = ""

    def get_context_from_retriver(self, query: str, db):
        return db.retrieve_using_similarity_search(query, top_k=10)

    def get_prompt(self, query: str, context: str) -> str:
        prompt = PromptTemplate(input_variables=["query", "context"], template=self.template)
        return prompt.format(query=query, context=context)


class FinalAPISelector(ReverseChainBaseClass):
    def __init__(self, model: str, temperature: float) -> None:
        super(FinalAPISelector, self).__init__(model, temperature)
        self.template = """
        We have below APIs that are similar to query:
        =====
        {context}
        =====
        If someone is saying: "{query}"
        Search for words like summarize, prioritize, my id, current sprint and select the api according to that.
        Try to divide the query in smaller tasks just like a human would do,
        now which final API should we use for this instruction? if no api can be used just return None in answer
        Only return API name in answer, donot return anything else.
        return the answer as a json object where key is api_name and key is the api name and a key data_source and value as the source of the file.
        Never give argument_name as the api name. 
        """

    def select_api_from_query(self, query: str, db) -> str:
        context = self.get_context_from_retriver(query, db)
        prompt = self.get_prompt(query, context=context)
        response = self.llm(prompt)
        return response


class ArgumentExtractor(ReverseChainBaseClass):
    def __init__(self, model: str, temperature: float) -> None:
        super(ArgumentExtractor, self).__init__(model, temperature)
        self.template = """
        You are an argument extractor. For each argument, you need to
        determine whether you can extract the value from user input
        directly or you need to use an API to get the value. The output
        should be in Json format, key is the argument, and value is the
        value of argument. Importantly, return None if you cannot get
        value.
        Give arguments that can be given to the API in context, if not found an 
        arguments value in the query return None.
        The api documentation is as below, use the context of the API to extract
        arguments that can be extracted from the user input and feeded in the API.
        if API doesnot use any arguments then, just return an empty json object:
        
        Context:
        {context}
        ......
        Now, Let's start.
        =>
        If someone is saying: "{query}"
        IMPORTANT:
        Donot try to make arguments if they are not present in the query, just return Null in place of the value.
        if the query contains key words like current sprint then use get_sprint_id
        if the query contains key words like my id them use who_am_is
        Arguments :
        """

    def get_arguments_from_query(self, query: str, db, api_documentation):
        prompt = self.get_prompt(query, api_documentation)
        response = self.llm(prompt)
        return response
    
class SubAPISelector(ReverseChainBaseClass):
    def __init__(self, model: str, temperature: float) -> None:
        super().__init__(model, temperature)
        self.template = """
        Required argument: {required_argument} 
        context: {context}
        Given the context above, give the API name that can give the reqiured_argument as output.
        if no api can be used just return None in answer
        Only return API name in answer, donot return anything else.
        return the answer as a json object where key is api_name and key is the api name and a key data_source and value as the source of the file.
        """

    def get_api_from_argument(self, db, required_argument: str) -> str:
        context = self.get_context_from_retriver(required_argument, db)
        prompt = self.get_prompt(context=context, required_argument=required_argument)
        response = self.llm(prompt)
        return response
    
    def get_context_from_retriver(self, query: str, db):
        return db.retrieve_using_similarity_search(query, top_k=5)

    def get_prompt(self, context: str, required_argument: str) -> str:
        prompt = PromptTemplate(input_variables=["context", "required_argument"], template=self.template)
        return prompt.format(context=context, required_argument=required_argument) 



# if __name__ == "__main__":
#     api_selector = APISelection(MODEL, TEMPERATURE)
#     argument_extractor = ArgumentExtractor(MODEL, TEMPERATURE)

#     query = "Please help Jack book a meeting room for 9:00-10:00"
#     selected_api = api_selector.select_api_from_query(query, db)
#     extracted_arguments = argument_extractor.get_arguments_from_query(query, db)

#     print(selected_api)
#     print(extracted_arguments)
