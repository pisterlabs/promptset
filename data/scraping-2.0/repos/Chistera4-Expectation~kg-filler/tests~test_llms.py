import tests
import unittest
from kgfiller.ai import ai_query
import kgfiller.ai.openai as openai
from kgfiller.ai.openai import OpenAiQuery
from kgfiller.ai.hugging import HuggingAiQuery
from kgfiller.ai.anthropic import AnthropicAiQuery


class TestLlms(unittest.TestCase):

    def llama(self):
        openai.almmai_endpoint()
        question = "Ingredients for baguette. names only."
        query = ai_query(question=question, api=OpenAiQuery)
        print(query.result_text)
        print(query.result_to_list())

    def hugging_falcon(self):
        question = "Ingredients for baguette. names only."
        query = ai_query(model='hugging_falcon', question=question, api=HuggingAiQuery)
        print(query.result_text)
        print(query.result_to_list())

    def hugging_mistral(self):
        question = "Ingredients for baguette. give names only."
        query = ai_query(model='hugging_mistral', question=question, api=HuggingAiQuery)
        print(query.result_text)
        print(query.result_to_list())

    def hugging_openchat(self):
        question = "Ingredients for baguette. names only."
        query = ai_query(model='hugging_openchat', question=question, api=HuggingAiQuery)
        print(query.result_text)
        print(query.result_to_list())

    def claude(self):
        question = "Ingredients for baguette. names only."
        query = ai_query(model='claude-instant-1', question=question, api=AnthropicAiQuery)
        print(query.result_text)
        print(query.result_to_list())