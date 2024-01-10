import json
import logging
import requests
from _decimal import Decimal
from typing import Optional, Dict, List, Union

import openai
from openai.openai_object import OpenAIObject

import replicate
from replicate import Client as rc

import sqlglot
from datasets import DatasetDict, Dataset

# from .helper import Prompts
from helper import Prompts

logger = logging.getLogger(__name__)

class SQLPredict: 
    """This class handles the dispatching of inference requests to various models. 
    """

    def __init__(
        self, 
        openai_api_key: str,
        replicate_api_key: str,
        hugging_face_api_key: Optional[str] = None,
    ) -> None:
        """Initialize the class"""

        openai.api_key = openai_api_key

        self.openai = openai
        self.prompts = Prompts()
        self.rc = rc(replicate_api_key)
        self.hf_key = hugging_face_api_key

        self.replicate_models = {}
        self.openai_api_models = {}

        self.model_endpoints = {}

    @classmethod
    def from_replicate_model(
        cls,
        openai_api_key: str,
        replicate_api_key: str,
        model_name: str,
        model_id: str,
    ) -> "SQLPredict":
        """Initialize the class with a Replicate model
        
        :param openai_api_key: The OpenAI API key.
        :type openai_api_key: str
        :param replicate_api_key: The Replicate API key.
        :type replicate_api_key: str
        :param model_name: The name of the Replicate model.
        :type model_name: str
        :param model_id: The ID of the Replicate model.
        :type model_id: str
        :return: The initialized class.
        :rtype: SQLPredict
        """
        
        instance = cls(openai_api_key, replicate_api_key)
        instance.replicate_models[model_name] = model_id

        return instance

    def __repr__(self):
        items = ("{}={!r}".format(k, self.__dict__[k]) for k in self.__dict__)
        return "{}({})".format(type(self).__name__, ", ".join(items))
    
    #########################################
    # Class Methods                         #
    #########################################

    def add_replicate_model(
        self,
        model_name: str,
        model_id: str,
    ) -> None:
        """Adds a Replicate model to the class.
        
        :param model_name: The name of the Replicate model.
        :type model_name: str
        :param model_id: The ID of the Replicate model.
        :type model_id: str
        """

        self.replicate_models[model_name] = model_id

    def add_model_endpoint(
        self,
        model_name: str,
        model_endpoint: str,
    ) -> None:
        """Adds a model endpoint to the class.
        
        :param model_name: The name of the model.
        :type model_name: str
        :param model_endpoint: The endpoint of the model.
        :type model_endpoint: str
        """

        self.model_endpoints[model_name] = model_endpoint

    #########################################
    # Request Construction Methods          #
    #########################################

    def _openai_sql_data_structure(
        self, 
        user_context: str,
        user_question: str,
        user_answer: str,
        system_context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Constructs a SQL data structure request for OpenAI's API.
        
        :param user_context: The context of the SQL query.
        :type user_context: str
        :param user_question: The question of the SQL query.
        :type user_question: str
        :param user_answer: The answer of the SQL query.
        :type user_answer: str
        :param system_context: The context of the SQL query, None results in class default
        :type system_context: Optional[str], optional
        :return: The constructed SQL data structure request.
        :rtype: List[Dict[str, str]]
        """
        
        if system_context is None:
            system_context = self.prompts._openai_sql_data_structure_prompt
        
        message = [
            {"role": "system", "content": system_context},
            {"role": "user", "content": f'Context: {user_context}\n\nQuestion": {user_question}\n\nAnswer: {user_answer}'},
        ]

        return message

    def _openai_sql_request_structure(
        self, 
        user_context: str,
        user_question: str,
        system_context: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Constructs a SQL request structure for OpenAI's API.

        :param user_context: The context of the SQL query.
        :type user_context: str
        :param user_question: The question of the SQL query.
        :type user_question: str
        :param system_context: The context of the SQL query, None results in class default
        :type system_context: Optional[str], optional
        :return: The constructed SQL request structure.
        :rtype: List[Dict[str, str]]
        """
        
        if system_context is None:
            system_context = self.prompts._openai_sql_request_structure_prompt
        
        message = [
            {"role": "system", "content": system_context},
            {"role": "user", "content": f'Context: {user_context}\n\nQuestion": {user_question}'},
        ]

        return message
    
    def openai_sql_response(
        self, 
        response_object: Union[OpenAIObject, Dict[str, str]],
        atl: Optional[bool] = False,
    ) -> Optional[str]: 
        """Parses the response from OpenAI's API.
        
        :param response_object: The response from OpenAI's API.
        :type response_object: OpenAIObject
        :return: The parsed response.
        :rtype: Optional[str]
        """

        if isinstance(response_object, OpenAIObject):
            response_object = response_object.to_dict()

        try:
            response = response_object['openai_inference']['choices'][0]['message']
        except Exception as e:
            logger.warning(f"OpenAI response failed to parse with error: {e}")
            return None

        if len(response.keys()) > 2: 
            logger.warning(f"OpenAI response has more than 2 keys: {response.keys()}")

        if atl:
            try: 
                sqlglot.parse(response["content"])
                return response["content"]
            except Exception as e:
                logger.warning(f"SQL query failed to parse with error: {e}")
                return None

        return response["content"]

    def openai_sql_request(
        self, 
        user_context: str,
        user_question: str,
        model: Optional[str] = "gpt-3.5-turbo", # TODO: consider using an enum for this
        system_context: Optional[str] = None,
        validate_response: Optional[bool] = False,
    ) -> Optional[OpenAIObject]:
        """Constructs a prompt to request a SQL query from OpenAI's API.
        
        :param user_context: The context of the SQL query.
        :type user_context: str
        :param user_question: The question of the SQL query.
        :type user_question: str
        :param model: The model to use for the request, defaults to "gpt-3.5-turbo"
        :type model: Optional[str], optional
        :param system_context: The context of the SQL query, None results in class default
        :type system_context: Optional[str], optional
        :param validate_response: Whether to validate the response, defaults to True. Returns None if validation fails.
        :type validate_response: Optional[bool], optional
        :return: The constructed SQL request.
        :rtype: OpenAIObject
        """

        message = self._openai_sql_request_structure(user_context, user_question, system_context)

        try: 
            request = self.openai.ChatCompletion.create(
                model=model, 
                messages=message,
            )
        except Exception as e:
            logger.warning(f"OpenAI request failed with error: {e}")
            raise e    

        if validate_response:
            return self.openai_sql_response(request)

        return request
    
    def openai_dataset_request(
        self, 
        dataset: Dataset,
    ): # -> Dict[str, OpenAIObject]: 
        """Constructs a prompt to request a SQL query from OpenAI's API.

        :param dataset: The dataset item to request.
        :type dataset: Dataset
        :return: The constructed SQL request.
        :rtype: OpenAIObject
        """
        try:
            context = dataset['context']
            question = dataset['question']
            inference = self.openai_sql_request(user_context=context, user_question=question)
        except Exception as e:
            logger.warning(f"OpenAI request failed with error: {e}")

        return {"openai_inference": inference}
    
    def replicate_sql_request(
        self, 
        prompt: str,
        model_name: str,
    ) -> str:
        """Constructs a prompt to request a SQL query from Replicate's API.

        :param prompt: The prompt to use for the request.
        :type prompt: str
        :return: The constructed SQL request.
        :rtype: str
        """
        
        try: 
            request = self.rc.run(
                self.replicate_models[model_name],
                input={"prompt": prompt},
            )
            return ''.join(item for item in request)
        except Exception as e:
            logger.warning(f"Replicate request failed with error: {e}")
            raise e    
        
    def replicate_dataset_request(
        self, 
        dataset: Dataset,
        model_name: Optional[str] = "llama_2_13b_sql",
        column_name: Optional[str] = "replicate_inference",
        prompt_type: Optional[str] = "tuning_format",
    ):
        """Constructs a prompt and requests a SQL query from Replicate's API.

        :param dataset: The dataset item to request.
        :type dataset: Dataset
        :return: The constructed SQL request.
        :rtype: str
        """

        if prompt_type == "tuning_format":
            prompt = json.loads(dataset['tuning_format'])['prompt']
        if prompt_type == "basic_text_generation":
            prompt = self.basic_text_generation_prompt(dataset['context'], dataset['question'])
        
        # assumes the prompt is in the dataset, contained within 'tuning_format'
        try:
            # prompt = json.loads(dataset['tuning_format'])['prompt']
            inference = self.replicate_sql_request(prompt, model_name=model_name)
            return {column_name: inference}
        except Exception as e:
            logger.warning(f"Replicate request failed with error: {e}")

    def basic_text_generation_prompt(
        self, 
        context: str,
        question: str,
    ) -> str: 
        """Constructs a basic text generation prompt.
        
        :param context: The context of the SQL query.
        :type context: str
        """

        prompt = "Context details the databse: " + context + " # " "Question to answer: " + question + " # " + "Answer as a SQL query: "
        return prompt

    def basic_text_generation_request(
        self, 
        context: str,
        question: str,
        model_name: str,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> str:
        """Constructs a basic text generation request.

        :param context: The context of the SQL query.
        :type context: str
        :param question: The question of the SQL query.
        :type question: str
        :param model_name: The name of the model.
        :type model_name: str
        :param api_key: The API key to use for the request, defaults to None. Defaults to class default.
        :type api_key: Optional[str], optional
        :param headers: The headers to use for the request, defaults to None. Defaults to class default.
        :type headers: Optional[Dict[str, str]], optional
        :return: The constructed SQL request.
        :rtype: str
        """
        
        if api_key is None:
            api_key = self.hf_key

        if headers is None:
            headers = {"Authorization": api_key}

        prompt = self.basic_text_generation_prompt(context, question)
        
        try: 
            response = requests.post(
                self.model_endpoints[model_name], 
                headers=headers,
                json={"inputs": prompt},
            )
            return response.json()
        except Exception as e:
            logger.warning(f"Basic text generation request failed with error: {e}")
            raise e
        
    def basic_text_generation_dataset_request(
        self, 
        dataset: Dataset,
        model_name: str,
        response_column_name: str,
        context_column_name: Optional[str] = "context",
        question_column_name: Optional[str] = "question",
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ):
        """Constructs a prompt and requests a SQL query from a generic API."""
        
        try:
            context = dataset[context_column_name]
            question = dataset[question_column_name]
            inference = self.basic_text_generation_request(context, question, model_name, api_key)
            return {response_column_name: inference}
        except Exception as e:
            logger.warning(f"Basic text generation request failed with error: {e}")