import json

import openai
from .agent import Agent
from search import bing_searcher

class ContextAgent(Agent):
    def __init__(self, api_key):
        super().__init__(api_key)
        self.bing_searcher = bing_searcher.BingSearcher()

    def query(self, context):
        self.context = context
        openai.api_key = self.api_key
        response = openai.ChatCompletion.create(
            model = self.model,
            messages=self.context.get_messages()
        )
        # print(self.context.get_messages())
        self.context.add_message("assistant", response["choices"][0]["message"]['content'])
        # print(response["choices"][0]["message"]['content'])
        return response["choices"][0]["message"]['content']

        # response = self.agent.run(prompt)
        # return response

    def query_with_functions(self, context):
        self.context = context
        openai.api_key = self.api_key
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.context.get_messages()
        )

        functions = [
            {
                "name": "general_search",
                "description": "Search the web for the given query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The bing web search query, e.g. Donald Trump signed a bill for trading "
                                           "with China.",
                        }
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "knowledge_graph_search",
                "description": "Search the bing knowledge graph for fact checking with a given query",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The bing knowledge graph search query, e.g. Donald Trump age.",
                        }
                    },
                    "required": ["query"],
                },
            }
        ]
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613",
            messages=context.get_messages(),
            functions=functions,
            function_call="auto",  # auto is default, but we'll be explicit
        )

        response_message = response["choices"][0]["message"]

        # print(response_message)

        if response_message.get("function_call"):
            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            available_functions = {
                "general_search": self.bing_searcher.general_search,
                "knowledge_graph_search": self.bing_searcher.knowledge_graph_search
            }  # only one function in this example, but you can have multiple
            function_name = response_message["function_call"]["name"]
            function_to_call = available_functions[function_name]
            function_args = json.loads(response_message["function_call"]["arguments"])
            function_response = function_to_call(
                query=function_args.get("query")
            )
            # print(function_response)
            # Step 4: send the info on the function call and function response to GPT
            context.add_message("system", str(response_message))
            context.add_function("function", function_name, str(function_response))
            second_response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-0613",
                messages=context.get_messages(),
            )  # get a new response from GPT where it can see the function response
            # print(second_response)
            context.add_message("assistant",second_response["choices"][0]["message"])
            return second_response["choices"][0]["message"]['content']

