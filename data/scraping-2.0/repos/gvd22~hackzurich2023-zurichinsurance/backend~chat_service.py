import inspect

import openai
import json
import os
from datetime import datetime
from qdrant_client import QdrantClient
from langchain.embeddings import OpenAIEmbeddings
from ticket_service import TicketServiceInstance


def current_date_format():
    jetzt = datetime.now()
    return jetzt.strftime('%A, %B %d, %Y %H:%M')


def check_args(function, args):
    sig = inspect.signature(function)
    params = sig.parameters

    # Check if there are extra arguments
    for name in args:
        if name not in params:
            return False
    # Check if the required arguments are provided
    for name, param in params.items():
        if param.default is param.empty and name not in args:
            return False
    return True


functions = [
    {
        "name": "get_current_status_of_claim",
        "description": "Get prior claim from the user by name",
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_name": {
                    "type": "string",
                },
            },
            "required": ["ticket_name"],
        },
    },
    {
        "name": "get_data_from_qdrant",
        "description": "Fetch data from the vector database",
        "parameters": {
            "type": "object",
            "properties": {
                "search_question": {
                    "type": "string",
                },
            },
            "required": ["search_question"],
        },
    }
]


class ChatService:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        openai.api_key = self.openai_api_key
        self.qdrant_url = "https://hackzurich23-vectordb-emcg5a6iia-oa.a.run.app"
        self.qdrant_port = 443
        self.qdrant_client = QdrantClient(location=self.qdrant_url, port=self.qdrant_port)
        self.embeddings = OpenAIEmbeddings(openai_api_key=self.openai_api_key)
        self.date_and_time = current_date_format()
        self.ticket_service = TicketServiceInstance
        self.available_functions = {
            "get_current_status_of_claim": self.ticket_service.get_ticket_by_name,
            "get_data_from_qdrant": self.qdrant_search_api,
        }

    def qdrant_search_api(self, search_question):
        embedded_query = self.embeddings.embed_query(search_question)
        hits = self.qdrant_client.search(
            collection_name="zurizap",
            query_vector=embedded_query,
            limit=10  # Return 5 closest points
        )
        prompt = ""
        for result in hits:
            prompt += result.payload['text']

        return " ".join([prompt])

    def run_conversation(self, user_input, messages):
        context = self.qdrant_search_api(user_input)

        if not messages:
            messages.append({"role": "system",
                 "content":
                     "You are a Customer Supporter for the Zurich Insurance and your Name ist ZüriZap."
                     "Your job is to answer questions about your client's insurance policies."
                     "You the AI were trained on data until 2021 and has no knowledge of events that have taken place since then. "
                     "You also have no way of accessing data on the internet, so you should not claim you can or say you will look. "
                     "Try to formulate your answers concisely, although this is not necessary. "
                     f"Closing date: Saturday, January 1, 2022 / Current date: {current_date_format()}"
                     "ZüriZap is the policy expert of the Zurich Insurance "
                     "ZüriZap never suggests to contact Zurich Insurance directly"
                     "ZüriZap never suggests to the user to read any of the documents provided by Zurich Insurance but provides the answer directly"
                     "All questions on this topic ZüriZap must always ground on the basis of information from the Knowledgebase."
                     "Answers should always be answered on the basis of the information provided to you. "
                     "It is important that the answer be short and precise."
                 })

        # Step 1: send the conversation and available functions to GPT
        messages.append({"role": "user", "content": f"You need to answer the question from the user: " + user_input +
                    "YOU SHOULD ALWAYS ANSWER BASED ON THIS INFORMATION IF YOU DONT HAVE ENOUGH INFORMATION THEN ASKE A QUESTION TO THE QDRANT DATABASE:" +
                    f"{context}"})

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0613", #"gpt-4-0613", # "gpt-3.5-turbo",
            messages=messages,
            functions=functions,
            function_call="auto",  # auto is default, but we'll be explicit
        )
        response_message = response["choices"][0]["message"]

        # Step 2: check if GPT wanted to call a function
        if response_message.get("function_call"):
            # Step 3: call the function
            # Note: the JSON response may not always be valid; be sure to handle errors
            # only one function in this example, but you can have multiple
            function_name = response_message["function_call"]["name"]

            if function_name not in self.available_functions:
                return "Function " + function_name + " does not exist", messages
            function_to_call = self.available_functions[function_name]

            # verify function has correct number of arguments
            function_args = json.loads(response_message["function_call"]["arguments"])
            if check_args(function_to_call, function_args) is False:
                return "Invalid number of arguments for function: " + function_name, messages
            function_response = function_to_call(**function_args)

            # Step 4: send the info on the function call and function response to GPT
            messages.append(response_message)  # extend conversation with assistant's reply
            messages.append(
                {
                    "role": "function",
                    "name": function_name,
                    "content": function_response,
                }
            )  # extend conversation with function response
            second_response = openai.ChatCompletion.create(
                model="gpt-4-0613", #"gpt-3.5-turbo-0613",
                messages=messages,
            )  # get a new response from GPT where it can see the function response

            response = second_response['choices'][0]['message']['content']
            messages.append(
                {
                    "role": "assistant",
                    "content": response,
                }
            )
            return response, messages
        else:
            chat_message = response['choices'][0]['message']['content']

            messages.append(
                {"role": "user", "content": f"You still nee to answer the question from the user: " + user_input +
                                            "Dont tell to the user that he need to contact Zurich Insurance and dont suggest to the user to read the insurace policies himself"
                                            "The following message is the current answer just change it when it doesnt refer to the rules i gave you"
                                            "You should'nt ask te user for a policy number because you already have it"
                                            f"{chat_message}"})


            response = openai.ChatCompletion.create(
                model="gpt-4-0613",  # "gpt-4-0613", # "gpt-3.5-turbo",
                messages=messages,

            )

            messages.append(
                {
                    "role": "assistant",
                    "content": response['choices'][0]['message']['content'],
                }
            )
        return chat_message, messages
