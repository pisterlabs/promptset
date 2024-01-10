import os # for environment variables
import json # for reading from data files

# openapi imports
import openai
import tiktoken
openai.api_key = os.getenv("OPENAI_API_KEY")

# define function classes for the api

class API: # class for functionality of the api
    def __init__(self):
        self.model = 'gpt-3.5-turbo'
        # create the message list with system messages from data/messages.json
        with open('data/messages.json', 'r') as file:
            json_data = file.read()
        self.messages = json.loads(json_data)

    def generate_response(self, prompt:str, max_tokens:int=300, temperature:float=1.0) -> str: # return response text
        # generate a OpenAI response from the prompt and update the corresponding variables
        self.add_message("user", prompt)

        response = openai.ChatCompletion.create(
            model=self.model, # gpt-3.5-turbo
            messages=self.messages, # prior messages
            temperature=temperature, # creativity scale
            max_tokens=max_tokens, # max tokens the API can use in its response
        )['choices'][0]['message']['content'] # get the string response from the API

        self.add_message("assistant", response)

        return response