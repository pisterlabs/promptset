from unittest import TestCase
from typing import AnyStr


class TestOpenAIFunctionAgent(TestCase):

    def test_query_from_agent(self):
        prompt = "List names, specialty and availability of all the contractors which " \
                 "location = San Jose. Display the {list} as HTML table"
        TestOpenAIFunctionAgent.__simple_query_contractor_test(prompt)

    def test_list_from_agent(self):
        prompt = "List all the contractors"
        TestOpenAIFunctionAgent.__list_contractor_test(prompt)


    @staticmethod
    def __list_contractor_test(prompt: AnyStr):
        from src.llm_langchain.openaifunctionagent import OpenAIFunctionAgent
        from test.domain.listentities import ListEntities

        tools = [ListEntities()]
        open_ai_function_agent = OpenAIFunctionAgent("gpt-3.5-turbo-0613", tools)
        answer = open_ai_function_agent(prompt)
        print(answer)

    @staticmethod
    def __simple_query_contractor_test(prompt: AnyStr):
        from src.llm_langchain.openaifunctionagent import OpenAIFunctionAgent
        from test.domain.simplequeryentities import SimpleQueryEntities

        tools = [SimpleQueryEntities()]
        open_ai_function_agent = OpenAIFunctionAgent("gpt-3.5-turbo-0613", tools)
        answer = open_ai_function_agent(prompt)
        print(answer)