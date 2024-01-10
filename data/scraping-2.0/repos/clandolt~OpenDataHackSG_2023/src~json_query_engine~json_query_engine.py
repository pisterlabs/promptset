import os
import openai
from llama_index.indices.service_context import ServiceContext
from llama_index.llms import OpenAI
from llama_index.indices.struct_store import JSONQueryEngine


class QueryEngine:
    """
    This class handles encodes the Text Query Prompt and queries the JSON-File

    ...

    Attributes
    ----------
    nl_query_engine : JSONQueryEngine
    raw_query_engine : JSONQueryEngine

    Methods
    -------
    get_nl_response(query : str)
    get_raw_response(query : str)

    """
    def __init__(self, open_api_key : str, json_value : dict, json_schema : dict):
        """
        Parameters
        ----------
        open_api_key : str
            Key to access the OpenAI API
        json_value : dict
            A JSON file with values as dict
        json_schema : dict
            A JSON file with the schema to query the data
        """
        self._observers = []
        os.environ["OPENAI_API_KEY"] = open_api_key
        openai.api_key = os.environ["OPENAI_API_KEY"]

        llm = OpenAI(model="gpt-3.5-turbo")
        service_context = ServiceContext.from_defaults(llm=llm)
        # a query engine which returns natural language answers
        self.nl_query_engine = JSONQueryEngine(
            json_value=json_value,
            json_schema=json_schema,
            service_context=service_context,
        )
        # a query engine which returns JSON data as answer
        self.raw_query_engine = JSONQueryEngine(
            json_value=json_value,
            json_schema=json_schema,
            service_context=service_context,
            synthesize_response=False,
        )

    def get_nl_response(self, query : str) -> str:
        """
        This method queries the JSON-File and returns the answer as natural language text.

        Parameters
        ----------
        query : str
            The query text as string

        Returns
        -------
        str
            The answer to the question as string
        """
        return self.nl_query_engine.query(query,)

    def get_raw_response(self, query : str) -> dict:
        """
        This method queries the JSON-File and returns the answer as JSON-structured data.

        Parameters
        ----------
        query : str
            The query text as string

        Returns
        -------
        dict
            The answer as JSON answer
        """
        return self.raw_query_engine.query(query,)
    
    def attach(self, observer):
        self._observers.append(observer)

    def detach(self, observer):
        self._observers.remove(observer)

    def notify(self, answer):
        for observer in self._observers:
            observer.update(answer)   

