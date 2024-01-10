
import server_logging
import pickle
import json
import config
import traceback
import os
import requests

from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


class MemoryManager:

    def __init__(self, frappe_base_url, api_key, api_secret):
        self.logger = server_logging.logging.getLogger('SERVER-MEMORY-MANAGER') 
        self.logger.addHandler(server_logging.file_handler)
        self.logger.info(f"Init MemoryManager")
        self.frappe_base_url = frappe_base_url
        self.headers = {
            'Authorization': f'token {api_key}:{api_secret}',
            'Content-Type': 'application/json'
        }

    def _send_request(self, method, endpoint, data=None):
            url = f"{self.frappe_base_url}{endpoint}"
            response = requests.request(method, url, headers=self.headers, json=data)
            return response.json()

    # Write Memory
    # how do we know when to create a new memory or update existing?
    # every time the human sends a message we create a new one if the exact request doesnt already exist

    def new_memory(self, query):
        query = self.clean_query(query)
        if self.get_memory(query):
            return True
        self.logger.debug(f"Adding new memory {query}")

        endpoint = '/resource/Bot%20Memory'
        data = {
            'query': query
            
        }
        response = self._send_request('POST', endpoint, data)

        self.logger.debug(str(response))
        if response.get('message') == 'Memory created':
            return True
        return True

    def clean_query(self, query):
        
        query = query[:400]
        query = query.replace("'", "")
        query = query.replace('"', "")
        query = query.replace(":", "")
        query = query.replace(";", "")
        query = query.replace("=", "")
        query = query.replace("/", "")
        query = query.replace("\\", "")
        
        return query


    def get_memory(self, query):
        query = self.clean_query(query)
        self.logger.info(f"Searching memory for {query}")
        endpoint = f'/resource/Bot%20Memory?filters=[["query","like","%{query}%"]]'
        response = self._send_request('GET', endpoint)
        self.logger.info(f"Response: {response}")
        
        # Check if 'data' is in the response and if it has at least one item
        if 'data' in response and response['data']:
            # Try to return the 'name' key from the first item in 'data'
            return response['data'][0].get('name', None)
        else:
            # Return None if 'data' is not in the response or it's an empty list
            return None

    def update_memory(self, query, response, api_key):
        query = self.clean_query(query)
        response = self.clean_query(str(response))

        existing_memory = self.get_memory(query)
        
        if not existing_memory:
            self.logger.error(f"memory doesnt exist {existing_memory}")
            return False

        self.logger.info(f"Matching Query: {query} to: {existing_memory}")

        memory_id = existing_memory
        self.logger.info(f'Updating memory {memory_id}')

        endpoint = f'/resource/Bot%20Memory/{memory_id}'

        evaluation = self.evaluate(query, response, api_key, config.MAIN_AI)
        evaluation = self.clean_query(evaluation)
        # Update only the name (or any other attributes you want to update)
        data = {
            'result': response,
            'evaluation': evaluation
        }

        success = self._send_request('PUT', endpoint, data)
        self.logger.info(str(success))
        # if success.get('message') == 'Memory Updated':
        #     return evaluation
        
        return evaluation


    # The main dispatcher LLM response
    def evaluate(self, query: str, response: str, api_key: str, openai_model: str) -> str:        
        try:
            os.environ["OPENAI_API_KEY"] = api_key
            llm = ChatOpenAI(temperature=0, model_name=openai_model, verbose=config.VERBOSE)
            
            prompt = f"""You are a assistant that reviews other assistants work, its your job to provide an evaluation of how well the response from the assistant meets the needs of the users request. Respond with a pass or fail reason for the score.

            Request: {query}

            Response: {response}"""
            self.logger.info(prompt)      

            return llm([HumanMessage(content=prompt)]).content
        
        except Exception as e:
            self.logger.error(f"{e} \n {traceback.format_exc()}")