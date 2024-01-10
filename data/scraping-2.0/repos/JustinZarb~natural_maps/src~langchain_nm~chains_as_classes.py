import os
import json
import pandas as pd
import requests
from time import gmtime, strftime
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.chains import (
    TransformChain,
    LLMChain,
    SimpleSequentialChain,
    SequentialChain,
)

# I let chatGPT convert everything into a class, but I feel I did something stupid..

# To use the OverpassQuery class, you can instantiate an object
# and then call the process_user_input method with the user's
# text input. For example:

# api_key = os.getenv("OPENAI_KEY")
# overpass_query = OverpassQuery(api_key)
# user_input = "Find bike parking near tech parks in Kreuzberg, Berlin."
# result = overpass_query.process_user_input(user_input)
# print(result)


class OverpassQuery:
    def __init__(self, api_key):
        self.api_key = api_key
        self.llm = ChatOpenAI(
            temperature=0, openai_api_key=self.api_key, model_name="gpt-3.5-turbo-0613"
        )
        self.chain_to_overpass_prompt = PromptTemplate(
            input_variables=["user_text_input"],
            template="""Turn the user's message into an overpass QL query.
            Example prompt: "Find bike parking near tech parks in Kreuzberg, Berlin.":\n\n {user_text_input}""",
        )
        self.chain_to_overpass = LLMChain(
            llm=self.llm, prompt=self.chain_to_overpass_prompt, output_key="ql_query"
        )
        self.perform_op_query_chain = TransformChain(
            input_variables=["ql_query"],
            output_variables=["overpass_answer"],
            transform=self.perform_op_query_func,
        )
        self.chain_to_user_prompt = PromptTemplate(
            input_variables=["overpass_answer", "user_text_input"],
            template="""Answer the user's message {user_text_input} based on the result of an overpass QL query contained in {overpass_answer}.""",
        )
        self.chain_to_user = LLMChain(llm=self.llm, prompt=self.chain_to_user_prompt)

        self.overpass_sequential_chain = SequentialChain(
            chains=[
                self.chain_to_overpass,
                self.perform_op_query_chain,
                self.chain_to_user,
            ],
            input_variables=["user_text_input"],
        )

    def overpass_query(self, ql_query):
        overpass_url = "http://overpass-api.de/api/interpreter"
        response = requests.get(overpass_url, params={"data": ql_query})
        if response.content:
            try:
                data = response.json()
            except:
                data = {"error": str(response)}
        else:
            print("Empty response from Overpass API")
            data = {
                "warning": "received an empty response from Overpass API. Tell the user."
            }
        data_str = json.dumps(data)
        return data_str

    def perform_op_query_func(self, inputs: dict) -> dict:
        query_input = inputs["ql_query"]
        op_answer = self.overpass_query(query_input)
        return {"overpass_answer": op_answer}

    def process_user_input(self, user_text_input):
        return self.overpass_sequential_chain.run({"user_text_input": user_text_input})
