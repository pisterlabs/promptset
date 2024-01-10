"""
api_calls.py
=====================
This module is part of the `abstract_ai` module of the `abstract_essentials` package. It provides essential utilities for making API calls to OpenAI, handling responses, and performing related operations.

Functions:
----------
- get_openai_key(): Fetches the OpenAI API key from the environment variables.
- load_openai_key(): Sets the API key for further OpenAI API interactions.
- headers(): Generates the headers for making API calls.
- post_request(): Sends a POST request to a specified endpoint.
- hard_request(): Sends a detailed request to OpenAI API.
- quick_request(): Quickly sends a request to OpenAI and fetches the response.
- get_additional_response(): Determines additional responses based on input.
- get_notation(): Determines notation or context notes.
- get_suggestions(): Suggests improvements or clarifications.
- get_abort(): Determines if a request should be aborted.
- get_default(): Returns the default response or user-defined response.
- default_instructions(): Generates a default instruction template.
- create_chunk_communication(): Formulates a communication format for chunked data.
- handle_abort(): Handles a scenario where a query might be aborted.
- process_response(): Converts the API response to a Python dictionary.
- get_current_chunk(): Retrieves current chunk data from a series.
- get_response_from_str(): Extracts response dictionary from a string representation.
- get_response(): Retrieves the JSON response from an API call.
- create_prompt_js(): Formulates the prompt in a JSON structure for API calls.
- get_save_output(): Saves the output of the API call.
- safe_send(): Safely sends a request to the OpenAI API and manages token limitations and chunked data.

Notes:
------
This module is intricately tied with environment variables and dependent modules. For seamless requests and response handling, ensure proper setup of dependencies and appropriate setting of environment variables.

About abstract_ai
--------------------
part of: abstract_ai
Version: 0.1.7.1
Author: putkoff
Contact: partners@abstractendeavors.com
Content Type: text/markdown
Source and Documentation:
For the source code, documentation, and more details, visit the official GitHub repository.
github: https://github.com/AbstractEndeavors/abstract_essentials/tree/main/abstract_ai

Notes:
------
This module relies heavily on environment variables and other dependencies to successfully send requests and handle responses. Ensure that you've set up all dependencies correctly and provided the right environment variables.
"""
import re
import os
import json
import openai
from abstract_utilities import (safe_json_loads,
                                safe_read_from_json,
                                json_key_or_default,
                                find_keys,
                                get_highest_value_obj,
                                get_files,
                                get_file_create_time,
                                eatAll)
from abstract_security.envy_it import get_env_value
from abstract_webtools import UrlManager
from .response_handling import ResponseManager
from .tokenization import calculate_token_distribution
from .prompts import default_prompt
from .endpoints import default_model,default_endpoint,default_tokens

class PromptManager:
    def __init__(self,prompt_data=None,
                 request=None,
                 instructions:str=None,
                 model:str=None,
                 title:str=None,
                 generate_title=True,
                 prompt_js:dict=None,
                 max_tokens:int=None,
                 completion_percentage:(int or float)=40,
                 endpoint:str=None,
                 content_type:str=None,
                 header=None,
                 api_env:str=None,
                 api_key:str=None,
                 additional_responses:bool=None,
                 directory:str=None,
                 abort = None,
                 notation=None,
                 suggestions=None,
                 role=None,
                 test_it=False):
        self.prompt_data=prompt_data
        self.model=model or default_model()
        self.max_tokens=max_tokens or default_tokens()
        self.endpoint= endpoint or default_endpoint()
        self.url_manager = UrlManager(url=self.endpoint)
        self.endpoint = self.url_manager.url
        self.request=request or "null"
        self.title=title or None
        self.generate_title=generate_title or self.title
        self.role=role or None
        self.content_type=content_type or 'application/json'
        self.api_env=api_env or 'OPENAI_API_KEY'
        self.api_key=api_key or self.get_openai_key()
        self.header=header or self.get_header()
        
        self.prompt_js=prompt_js or {}
        
        self.test_it=test_it
        self.completion_percentage=completion_percentage or 40
        self.additional_responses=additional_responses or False
        self.directory=directory or os.getcwd()
        self.response_data_directory=self.directory
        if os.path.basename(self.response_data_directory) != 'response_data':
             self.response_data_directory= os.path.join(self.directory,'response_data')
        self.bot_notation=''
        self.notation=notation or self.test_it
        self.abort=abort or self.test_it
        self.notation=notation or self.test_it
        self.suggestions=suggestions or self.test_it

        self.instructions= instructions
        self.chunk='01'
        self.total_chunks = '01'
        
        self.token_dist=None
        self.initialize()
        
        self.output=[]
        self.save_response=None
        self.abort_it=None
        self.additional_response_it=None
    def get_openai_key(self):
        """
        Retrieves the OpenAI API key from the environment variables.

        Args:
            key (str): The name of the environment variable containing the API key. 
                Defaults to 'OPENAI_API_KEY'.

        Returns:
            str: The OpenAI API key.
        """
        return get_env_value(key=self.api_env)
    def initialize(self):
        self.load_openai_key()
        self.get_instructions()
        self.token_dist=calculate_token_distribution(bot_notation=self.bot_notation,max_tokens=self.max_tokens,completion_percentage=self.completion_percentage,prompt_guide=self.create_prompt_guide(),chunk_prompt=self.prompt_data)
        self.chunk=0
        self.total_chunks = self.token_dist["chunks"]["total"]
    def get_instructions(self):
        if self.instructions:
            if not isinstance(self.instructions,dict):
                self.instructions={"instruction":self.instructions}
        else:
            self.instructions = {}
        
        if 'response' not in self.instructions:
            self.instructions["api_response"]="place response to prompt here"
        if self.notation or self.test_it:
           self.instructions["notation"]=self.get_notation()
        if self.suggestions or self.test_it:
            self.instructions["suggestions"]=self.get_suggestions()
        if self.additional_responses or self.test_it:
            self.instructions["additional_response"]=self.get_additional_response()
        if self.abort or self.test_it:
            self.instructions["abort"]=self.get_abort()
        if self.generate_title or self.test_it:
            self.instructions["title"]= self.get_title()
        self.get_instruction()
    def get_additional_response(self,):
        """
        Determines the additional response based on the input value.
        
        Args:
            bool_value (bool or str): Input value based on which the response is determined.
            
        Returns:
            str: The determined response.
        """
        if isinstance(self.additional_responses,str):
            return self.additional_responses
        if self.additional_responses:
            return "this value is to be returned as a bool value,this option is only offered to the module if the user has allowed a non finite completion token requirement. if your response is constrained by token allowance, return this as True and the same prompt will be looped and the responses collated until this value is False at which point the loop will ceased and promptng will resume once again"
        return "return false"
    def get_title(self):
        """
        Retrieves the notation based on the input value.
        
        Args:
            bool_value (bool or str): Input value based on which the notation is determined.
            
        Returns:
            str: The determined notation.
        """
        if isinstance(self.title,str):
            return self.title
        if self.generate_title:
            return 'please generate a title for this chat based on the both the context of the query and the context of your response'
        return "return false"
    def get_notation(self):
        """
        Retrieves the notation based on the input value.
        
        Args:
            bool_value (bool or str): Input value based on which the notation is determined.
            
        Returns:
            str: The determined notation.
        """
        if isinstance(self.notation,str):
            return self.notation
        if self.notation:
            return "insert any notes you would like to recieve upon the next chunk distribution in order to maintain context and proper continuity"
        return "return false"
    def get_suggestions(self):
        """
        Retrieves the suggestions based on the input value.
        
        Args:
            bool_value (bool or str): Input value based on which the suggestion is determined.
            
        Returns:
            str: The determined suggestions.
        """
        if isinstance(self.suggestions,str):
            return self.suggestions
        if self.suggestions:
            return "insert any suggestions you find such as correcting ambiguity in the prompts entirety such as context, clear direction, anything that will benefit your ability to perform the task"
        return "return false"
    def get_abort(self):
        """
        Retrieves the abort based on the input value.
        
        Args:
            bool_value (bool or str): Input value based on which the abort is determined.
            
        Returns:
            str: The determined abort.
        """
        if isinstance(self.abort,str):
            return self.abort
        if self.abort:
            return "if you cannot fullfil the request, return this value True; be sure to leave a notation detailing whythis was"
        return "return false"
    def get_header(self):
        """
        Generates request headers for API call.
        
        Args:
            content_type (str): Type of the content being sent in the request. Default is 'application/json'.
            api_key (str): The API key for authorization. By default, it retrieves the OpenAI API key.
            
        Returns:
            dict: Dictionary containing the 'Content-Type' and 'Authorization' headers.
        """
        return {'Content-Type': self.content_type, 'Authorization': f'Bearer {self.api_key}'}
    def load_openai_key(self):
        """
        Loads the OpenAI API key for authentication.
        """
        openai.api_key = self.api_key
    
    def get_for_prompt(self,title,data):
        if data:
            return f'#{title}#'+'\n\n'+f'{data}'
        return ''
    def get_instruction(self):
        self.instruction = "your response is expected to be in JSON format with the keys as follows:\n"
        if self.test_it:
            self.instruction += 'this query is a test, please place a test response in every key\n'
        self.instruction += '\n\n'
        for i,key in enumerate(self.instructions.keys()):
            self.instruction+=f"{i}) {key} - {self.instructions[key]}\n"
    def create_prompt_guide(self):
        """
        Creates a formatted communication for the current data chunk.

        Returns:
            str: The formatted chunk communication.
        """
        
        
        prompt_data_chunk=self.prompt_data or '' 
        if self.token_dist:
            prompt_data_chunk=self.token_dist["chunks"]["data"][self.chunk] or ''
        return f'''
                {self.get_for_prompt('data chunk',self.total_chunks)}
                {self.get_for_prompt('instructions',self.instruction)}
                {self.get_for_prompt('prompt',self.request)}
                {self.get_for_prompt('notation from the previous response',self.bot_notation)}
                {self.get_for_prompt('current data chunk',prompt_data_chunk)}
                '''
    def create_prompt(self):
        """
        Creates a prompt dictionary with the specified values.

        Args:
            js (dict, optional): The input JSON dictionary. Defaults to None.
            model (str, optional): The model name. Defaults to default_model().
            prompt (str, optional): The prompt string. Defaults to default_prompt().
            max_tokens (int, optional): The maximum number of tokens. Defaults to default_tokens().

        Returns:
            dict: The prompt dictionary.
        """
        self.prompt =""
        max_tokens = self.max_tokens
        if self.token_dist:
            self.prompt =self.create_prompt_guide()
            max_tokens = self.token_dist["completion"]["available"]
        return {"model": self.model, "messages": [{"role": self.role or "user", "content": self.prompt}],"max_tokens": max_tokens}
    def send_query(self):
        for self.chunk in range(self.total_chunks):
            response_loop=True
            self.abort_it=False
            while response_loop:
                self.prompt_js=self.create_prompt()
                self.response_handler = ResponseManager(endpoint=self.endpoint,prompt=self.prompt_js,header=self.header,title=self.title,directory=self.response_data_directory)
                self.query_js,self.api_response,self.abort_it,self.additional_response_it,self.bot_notation = self.response_handler.query_js,self.response_handler.api_response,self.response_handler.abort_it,self.response_handler.additional_response_it,self.response_handler.bot_notation
                self.output.append(self.query_js)
                if not self.additional_response_it or self.abort_it:
                    response_loop = False
                    break
            if self.abort_it:
                break
            self.chunk+=1
        return self.output
def hard_request(prompt: str = default_prompt(), model: str = default_model(), max_tokens: int = default_tokens(),
                 temperature: float = 0.5, top_p: int = 1, frequency_penalty: int = 0, presence_penalty: int = 0,api_key=None,api_env_key=None):
    """
    Sends a hard request to the OpenAI API using the provided parameters.

    Args:
        max_tokens (int): The maximum number of tokens for the completion.
        prompt (str): The prompt for the API request.

    Returns:
        dict: The response received from the OpenAI API.
    """
    if api_env_key == None:
        api_env_key = 'OPEN_AI_API'
    if api_key == None:
        api_key=get_env_value(key=api_env_key)
    openai.api_key = api_key
    message = {
        "role": "user",
        "content": prompt
    }
    response = openai.ChatCompletion.create(
        model=model,
        messages=[message]
    )
    return response
def quick_request(prompt:str=default_prompt(),max_tokens:int=default_tokens(),model=default_model(),api_key=None,api_env_key=None):
    """
    Sends a quick request to the OpenAI API using the provided parameters and prints the result.

    Args:
        prompt (str, optional): The prompt for the API request. Defaults to the default_prompt().
        max_tokens (int, optional): The maximum number of tokens for the completion. Defaults to default_tokens().

    Returns:
        None
    """

    return hard_request(max_tokens=max_tokens, prompt=prompt,model=model,api_key=api_key,api_env_key=api_env_key)
