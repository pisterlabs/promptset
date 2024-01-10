import yaml
import openai
import json
import streamlit as st
from secret import keys
from src.gpt_function import GPTFunction, gpt_agent
from data import core

class BaseAgent:
    def __init__(self, functions: list[GPTFunction]):
        openai.api_key = keys.openai_key
        config = yaml.safe_load(open("config.yaml", "r"))
        self.model_name = config["model"]["agent"]
        self.messages = []
        self.functions = {}
        self.max_retries = 3
        for function in functions:
            self.functions[function.name] = function

    def get_response(self, prompt: str, allow_function_calls: bool = True):
        print("\nSystem:")
        print(prompt)
        self.messages.append({"role": "system", "content": prompt})
        available_data = {}
        for name, data in core.get_all_data_details().items():
            available_data[name] = data["summary"]
        available_data = json.dumps(available_data, indent=4)
        data_message = [{"role": "system", "content": f"Data available from storage:\n{available_data}"}]
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=self.messages + data_message,
            functions=list(map(lambda x: x.to_dict(), self.functions.values())),
            function_call="auto" if allow_function_calls else "none"
        )["choices"][0]["message"]
        self.messages.append(response)

        if response.get("function_call") and allow_function_calls:
            func_name = response["function_call"]["name"]
            func_args = response["function_call"]["arguments"]
            func_args = json.loads(func_args)
            self.call_function(func_name, func_args)
            return None
        else:
            print("\nAgent:")
            print(response["content"])

        return response["content"]

    def call_function(self, func_name: str, func_args: dict):
        print("\nFunction call:\n", func_name, "\n", func_args)
        func = self.functions[func_name]
        func_results = func(func_args)
        print("\nFunction results:\n", func_results)
        self.messages.append({"role": "function", "name": func_name, "content": func_results})

    def run(self, task: str) -> str:
        pass
