"""
A module for interacting with the Language Learning Model
"""
import functools
import hashlib
import json
import logging
import os
import traceback
from abc import abstractmethod
from enum import Enum
from typing import Any

import openai
from tenacity import retry, wait_fixed, stop_after_attempt

from pydantic import BaseModel, Field

from func_ai.utils.common import arg_in_func
from func_ai.utils.py_function_parser import type_mapping, func_to_json

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


class ConversationStore(BaseModel):
    """
    A class for storing conversations

    """
    conversation: list = Field(default_factory=list, description="A dictionary of conversations")

    @abstractmethod
    def add_message(self, message: Any):
        """
        Adds a message to the conversation
        :param message:
        :return:
        """
        raise NotImplementedError

    def get_conversation(self) -> list:
        """
        Returns the conversation
        :return:
        """
        return self.conversation

    def get_last_message(self) -> Any:
        """
        Returns the last message
        :return:
        """
        return self.conversation[-1]

    def clear(self) -> None:
        """
        Clears the conversation
        :return:
        """
        self.conversation = []


class OpenAIConversationStore(ConversationStore):

    def add_message(self, message: Any):
        self.conversation.append(message)

    def add_system_message(self, message: str):
        # remove any existing system messages
        self.conversation = [x for x in self.conversation if x['role'] != 'system']
        self.conversation.insert(0, {"role": "system", "content": message})


class LLMInterface(BaseModel):
    """
    Interface for interacting with the Language Learning Model

    """
    usage: dict = Field(default_factory=dict, description="A dictionary of the usage of the API")
    cost_mapping: dict = Field(default_factory=dict, description="A dictionary of the cost of the API")
    conversation_store: ConversationStore = Field(...,
                                                  description="A class for storing conversations")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def send(self, prompt: str, **kwargs) -> Any:
        """
        Sends a prompt to the API

        :param prompt:
        :param kwargs:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def load_cost_mapping(self, file_path: str) -> None:
        """
        Loads the cost mapping from a file

        :param file_path:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def update_cost(self, model, api_response) -> None:
        """
        Updates the cost of the API

        :param model: The model used
        :param api_response: The response from the API
        :return:
        """
        raise NotImplementedError

    def get_usage(self) -> dict:
        return self.usage

    def reset_usage(self) -> None:
        """
        Resets the usage of the API

        :return:
        """
        self.usage = dict()

    def store_usage(self, file_path: str) -> None:
        """
        Appends the usage of the API to a file

        :param file_path: The path to the file
        :return:
        """
        with open(file_path, "a") as f:
            json.dump(self.usage, f)

    def get_conversation(self) -> list[any]:
        """
        Returns the conversation
        :return:
        """
        return self.conversation_store.get_conversation()

    def add_conversation_message(self, message: any, update_llm: bool = False, **kwargs) -> "LLMInterface":
        """
        Adds a message to the conversation
        :param message: The message to add
        :param update_llm: Whether to update the LLM with the new conversation
        :param kwargs: Parameters to pass to the API
        :return:
        """
        self.conversation_store.add_message(message)
        if update_llm:
            self.update_llm_conversation(**kwargs)
        return self

    def update_llm_conversation(self, **kwargs) -> "LLMInterface":
        """
        Sends the updated conversation to the LLM

        :param kwargs: Parameters to pass to the API
        :return:
        """
        raise NotImplementedError


class OpenAIInterface(LLMInterface):
    """
    Interface for interacting with the OpenAI API
    """
    max_tokens: int = Field(default=256, description="The maximum number of tokens to return")
    model: str = Field(default="gpt-3.5-turbo-0613", description="The model to use")
    temperature: float = Field(default=0.0, description="The temperature to use")
    conversation_store: OpenAIConversationStore = Field(default_factory=OpenAIConversationStore,
                                                        description="A class for storing conversations")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, max_tokens=kwargs.get("max_tokens", 256),
                         model=kwargs.get("model", "gpt-3.5-turbo-0613"),
                         temperature=kwargs.get("temperature", 0.0),
                         conversation_store=OpenAIConversationStore())
        openai.api_key = kwargs.get("api_key", os.getenv("OPENAI_API_KEY"))

    @retry(stop=stop_after_attempt(3), reraise=True, wait=wait_fixed(1),
           retry_error_callback=lambda x: logger.warning(x))
    def update_llm_conversation(self, **kwargs) -> "OpenAIInterface":
        """
        Sends the updated conversation to the LLM

        :param kwargs: Parameters to pass to the API
        :return:
        """
        _functions = kwargs.get("functions", None)
        _model = kwargs.get("model", self.model)
        try:
            if _functions:
                response = openai.ChatCompletion.create(
                    model=_model,
                    messages=self.conversation_store.get_conversation(),
                    functions=_functions,
                    function_call="auto",
                    temperature=kwargs.get("temperature", self.temperature),
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                )
            else:
                response = openai.ChatCompletion.create(
                    model=_model,
                    messages=self.conversation_store.get_conversation(),
                    temperature=kwargs.get("temperature", self.temperature),
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    max_tokens=kwargs.get("max_tokens", self.max_tokens),
                )
                # decode utf-8 bytes to unicode
            if "content" in response["choices"][0]["message"] and response["choices"][0]["message"]["content"]:
                response["choices"][0]["message"]["content"] = response["choices"][0]["message"]["content"].encode(
                    'utf-8').decode(
                    "utf-8")
            if "function_call" in response["choices"][0]["message"] and response["choices"][0]["message"][
                "function_call"]:
                response["choices"][0]["message"]["function_call"]["arguments"] = \
                    response["choices"][0]["message"]["function_call"]["arguments"].encode('utf-8').decode("utf-8")
            self.update_cost(_model, response)
            _response_message = response["choices"][0]["message"]
            self.conversation_store.add_message(_response_message)
            return self
        except Exception as e:
            logger.error(f"Error: {e}")
            traceback.print_exc()
            raise e

    def send(self, prompt: str, **kwargs) -> dict[str, any]:
        # print(type(self._conversation_store))
        self.conversation_store.add_message({"role": "user", "content": prompt})
        logger.debug(f"Prompt: {prompt}")
        response = self.update_llm_conversation(**kwargs)
        logger.debug(f"Response: {response}")
        return self.conversation_store.get_last_message()

    def load_cost_mapping(self, file_path: str) -> None:
        with open(file_path) as f:
            self.cost_mapping = json.load(f)

    def update_cost(self, model, api_response) -> None:
        if model not in self.usage:
            self.usage[model] = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        self.usage[model]["prompt_tokens"] += api_response['usage']['prompt_tokens']
        self.usage[model]["completion_tokens"] += api_response['usage']['completion_tokens']
        self.usage[model]["total_tokens"] += api_response['usage']['total_tokens']


class OpenAISchema(BaseModel):
    @classmethod
    @property
    def openai_schema(cls):
        schema = cls.schema()

        return {
            "name": schema["title"],
            "description": schema.get("description", f"{schema['title']} class"),
            "parameters": {
                "type": "object",
                "properties": {f"{name}": OpenAISchema.get_field_def(name, field_info) for name, field_info in
                               cls.__fields__.items()},
            },
            "required": [name for name, field_info in cls.__fields__.items() if field_info.required]
        }

    @staticmethod
    def get_field_def(name, field_info) -> dict[str, str]:
        """
        Returns a string representation of a field definition

        :param name:
        :param field_info:
        :return:
        """

        default = f". Default value: {str(field_info.default)}" if not field_info.required else ""
        description = field_info.field_info.description
        if description:
            description = description.replace("\n", " ")
        else:
            description = ""
        _enum_values = ""
        if issubclass(field_info.outer_type_, Enum):
            _enum_values = ". Enum: " + ",".join([f"{_enum.name}" for _enum in field_info.outer_type_])
        return {
            "description": f"{description}{default}{_enum_values}",
            "type": type_mapping(field_info.outer_type_)
        }

    @classmethod
    def from_response(cls, completion, throw_error=True):
        """
        Returns an instance of the class from LLM completion response

        :param completion: completion response from LLM
        :param throw_error:  whether to throw error if function call is not present
        :return:
        """
        if throw_error:
            assert "function_call" in completion, "No function call detected"
            assert (
                    completion["function_call"]["name"] == cls.openai_schema["name"]
            ), "Function name does not match"

        function_call = completion["function_call"]
        arguments = json.loads(function_call["arguments"])
        return cls(**arguments)

    @classmethod
    def from_prompt(cls, prompt: str, llm_interface: LLMInterface, throw_error=True):
        """
        Returns an instance of the class from LLM prompt

        :param prompt: User prompt
        :param llm_interface: LLM interface
        :param throw_error: whether to throw error if function call is not present
        :return:
        """
        completion = llm_interface.send(prompt, functions=[cls.openai_schema])
        # print(llm_interface.get_conversation())
        # TODO add crud interface functions here
        return cls.from_response(completion, throw_error)


class OpenAIFunctionWrapper(object):
    """
    A wrapper class for OpenAI functions.
    """

    def __init__(self, llm_interface: LLMInterface, name: str, description: str, parameters: dict[str, any],
                 func: callable, **kwargs) -> None:
        """
        Initializes the OpenAI function wrapper
        :param llm_interface:
        :param name:
        :param description:
        :param parameters:
        :param func:
        :param kwargs: All these are treated as metadata that can be used by other tooling to augment the function
        """
        self.llm_interface = llm_interface
        self._name = name
        self._description = description
        assert "required" in parameters, "Required field not present in parameters"
        self._parameters = parameters
        self.is_partial = False
        assert callable(func) or isinstance(func, functools.partial), "Function must be callable"
        if arg_in_func(func, "action"):
            self.func = functools.partial(func, action=self)
        else:
            self.func = func
        if isinstance(func, functools.partial):
            module_name = func.func.__module__
            function_name = func.func.__qualname__  # use __qualname__ for nested functions
            self.is_partial = True
        else:
            module_name = func.__module__
            function_name = func.__qualname__  # use __qualname__ for nested functions
        self._identifier = f"{module_name}.{function_name}"
        if "<lambda>" in self._identifier:
            raise ValueError("Cannot use lambda functions")
        if "<locals>" in self._identifier:
            raise ValueError("Cannot use nested functions")
        self._hash = hashlib.sha1(f"{self._identifier}-{self._name}-{self._description}".encode()).hexdigest()
        self._metadata = kwargs
        self._llm_calls = []

    @property
    def identifier(self) -> str:
        """
        Returns the identifier of the function

        :return: identifier of the function
        """
        return self._identifier

    @property
    def hash(self) -> str:
        """
        Returns the identifier of the function

        :return: identifier of the function
        """
        return self._hash

    @property
    def name(self) -> str:
        """
        Returns the name of the function

        :return: name of the function
        """
        return self._name

    @property
    def description(self) -> str:
        """
        Returns the description of the function

        :return: description of the function
        """
        return self._description

    @property
    def parameters(self) -> dict[str, any]:
        """
        Returns the parameters of the function

        :return: parameters of the function
        """
        return self._parameters

    def call(self, *args, **kwargs) -> dict[str, any]:
        """
        Calls the function with the given arguments

        :param kwargs: arguments to be passed to the function
        :return:
        """
        return self.func(*args, **kwargs)

    @property
    def metadata(self) -> dict[str, any]:
        """
        Returns the metadata of the function

        :return: dict containing the metadata
        """
        return self._metadata

    @property
    def metadata_dict(self) -> dict[str, str]:
        """
        Returns the metadata of the function as a

        :return:
        """
        return {k: str(v) for k, v in self._metadata.items()}

    @property
    def schema(self) -> dict[str, any]:
        """
        Returns the schema of the function that can be passed to OpenAI API

        :return: dict containing the schema
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def __str__(self) -> str:
        return f"{self.schema}"

    def __repr__(self) -> str:
        return f"{self.schema}"

    def to_dict(self) -> dict[str, any]:
        """
        Returns a dict representation of the function

        :return: dict containing the function
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
            "metadata": self.metadata,
        }

    @property
    def last_call(self) -> dict[str, any]:
        """
        Returns the last response from the function call

        :return: dict containing the last response
        """
        return self._llm_calls[-1]

    @property
    def calls(self) -> list[dict[str, any]]:
        """
        Returns the list of responses from the function call

        :return: list containing the responses
        """
        return self._llm_calls

    def from_response(self, llm_response: dict[str, any]) -> "OpenAIFunctionWrapper":
        """
        Returns an instance of the class from LLM completion response

        :param llm_response: completion response from LLM
        :return: The response from the function call
        """
        if "function_call" not in llm_response:
            raise ValueError(f"No function call detected: {llm_response}")
        if llm_response["function_call"]["name"] != self.name:
            raise ValueError(f"Function name does not match: {llm_response}")
        try:
            _func_response = self.func(**json.loads(llm_response["function_call"]["arguments"]))
        except Exception as e:
            _func_response = f"Error: {repr(e)}"
            logger.warning(f"Failed to process function call: {llm_response}")
            traceback.print_exc()
        _function_call_llm_response = {
            "role": "function",
            "name": self.name,
            "content": f"{_func_response}",
        }
        self._llm_calls.append({
            "function_call": llm_response["function_call"],
            "function_response": _function_call_llm_response,
        })
        self.llm_interface.add_conversation_message(_function_call_llm_response)
        return self

    def from_response_raw(self, llm_response: dict[str, any]) -> any:
        """
        Returns an instance of the class from LLM completion response

        :param llm_response: completion response from LLM
        :return: The response from the function call
        """
        if "function_call" not in llm_response:
            raise ValueError(f"No function call detected: {llm_response}")
        if llm_response["function_call"]["name"] != self.name:
            raise ValueError(f"Function name does not match: {llm_response}")
        try:
            _func_response = self.func(**json.loads(llm_response["function_call"]["arguments"]))
            return _func_response
        except Exception as e:
            logger.warning(f"Failed to process function call: {llm_response}")
            raise e

    def from_prompt(self, prompt: str, **kwargs) -> "OpenAIFunctionWrapper":
        """
        Returns an instance of the class from LLM prompt

        :param prompt: User prompt
        :param kwargs: arguments to be passed to the function
        :return: The response from the function call
        """
        self.from_response(self.llm_interface.send(prompt, functions=[self.schema]))
        return self

    @classmethod
    def from_python_function(cls, func: callable, llm_interface: LLMInterface, **kwargs) -> "OpenAIFunctionWrapper":
        """
        Returns an instance of the class from Python function

        :param func: Python function
        :param llm_interface: LLM interface
        :param kwargs: arguments to be passed to the function
        :return:
        """
        _func = func_to_json(func)
        return cls(llm_interface=llm_interface, func=func, **_func, **kwargs)
