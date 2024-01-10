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


class OverpassQueryChain:
    def __init__(self, api_key):
        self.api_key = api_key
        self.llm = ChatOpenAI(
            temperature=0.1,
            openai_api_key=self.api_key,
            model_name="gpt-3.5-turbo-0613",
        )
        self.overpass_answer = None
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
            template="""
            Answer the user's message {user_text_input} based on the result of an overpass QL query contained in {overpass_answer}.
            In your answer, don't mention overpass QL. In your answer, don't mention latitude or longitude, but rather use street addresses.
            """,
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
        self.messages = []
        self.overpass_queries = {}

    def save_to_json(self, file_path: str, timestamp: str, log: dict):
        json_file_path = file_path

        # Check if the folder exists and if not, create it.
        folder_path = os.path.dirname(json_file_path)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Check if the file exists
        if os.path.isfile(json_file_path):
            # If it exists, open it and load the JSON data
            with open(json_file_path, "r") as f:
                data = json.load(f)
        else:
            # If it doesn't exist, create an empty dictionary
            data = {}

        # Add data for this run
        this_run_name = f"{timestamp}"
        data[this_run_name] = {
            "log": log,
        }

        with open(json_file_path, "w") as f:
            json.dump(data, f, indent=4)

    def get_timestamp(self):
        return strftime("%Y-%m-%d %H:%M:%S", gmtime())

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
        # Write Overpass API Call to JSON
        timestamp = self.get_timestamp()
        filepath = os.path.expanduser("./naturalmaps_logs/overpass_query_log.json")
        success = True if "error" not in data_str else False
        self.overpass_queries[ql_query] = {
            "success": success,
            "returned something": "I don't know",
            "data": data_str,
        }
        self.save_to_json(
            file_path=filepath,
            timestamp=timestamp,
            log={
                "query": ql_query,
                "response": data_str,
                "query_success": success,
                "returned_something": "I don't know",
            },
        )

        return data_str

    def perform_op_query_func(self, inputs: dict) -> dict:
        query_input = inputs["ql_query"]
        op_answer = self.overpass_query(query_input)
        self.overpass_answer = op_answer
        return {"overpass_answer": op_answer}

    def process_user_input(self, user_text_input):
        return self.overpass_sequential_chain.run({"user_text_input": user_text_input})


# tests_b = []

# inputs = pd.read_csv("./dev/prompts/prompts.csv")


# def add_to_tests():
#     i = len(tests_b)
#     output = overpass_query.process_user_input(inputs[i])
#     tests_b.append(output)
