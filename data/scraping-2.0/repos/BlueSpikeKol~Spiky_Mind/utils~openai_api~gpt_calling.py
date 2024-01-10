import inspect
from typing import Optional, Union, List
from dataclasses import dataclass
import time

import openai

import utils.openai_api.templates as templates
import utils.openai_api.models as models
import utils.openai_api.token_pools as token_pools
from utils.openai_api.models import ModelType, ModelManager
from utils import config_retrieval

EXAMPLE_CHAT_COMPLETION = """
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "gpt-3.5-turbo-0613",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "\n\nHello there, how may I assist you today?",
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 9,
    "completion_tokens": 12,
    "total_tokens": 21
  }
}
"""
EXAMPLE_TEXT_COMPLETION = """
{
  "id": "cmpl-uqkvlQyYK7bGYrRHQ0eXlWi7",
  "object": "text_completion",
  "created": 1589478378,
  "model": "gpt-3.5-turbo",
  "choices": [
    {
      "text": "\n\nThis is indeed a test",
      "index": 0,
      "logprobs": null,
      "finish_reason": "length"
    }
  ],
  "usage": {
    "prompt_tokens": 5,
    "completion_tokens": 7,
    "total_tokens": 12
  }
}
"""


class GPTAgentHandler:
    def __init__(self, agent):
        self.agent = agent

    def prepare_messages(self):
        if isinstance(self.agent.messages, list):
            messages_copy = self.agent.messages.copy()
        elif isinstance(self.agent.messages, dict):
            messages_copy = [self.agent.messages]
        elif isinstance(self.agent.messages, str):
            messages_copy = [{'role': 'user', 'content': self.agent.messages}]
        else:
            messages_copy = []

        if self.agent.system_prompt:
            if isinstance(self.agent.system_prompt, str):
                system_prompt_dict = {'role': 'system', 'content': self.agent.system_prompt}
                messages_copy.insert(0, system_prompt_dict)
            else:
                messages_copy.insert(0, self.agent.system_prompt)
        else:
            default_system_prompt = {'role': 'system', 'content': 'you are a helpful AI assistant'}
            messages_copy.insert(0, default_system_prompt)

        chat_message_object = templates.TemplateManager.transform_into_messages(messages_copy)
        prompt_string = " ".join([message_dict['content'] for message_dict in chat_message_object.messages])
        ModelManager.check_agent_token_limit(model=self.agent.model, prompt=prompt_string,
                                             max_tokens=self.agent.max_tokens)
        # TODO make sure that the inputed message is a collection of the system prompt and previous messages
        return {'messages': chat_message_object.messages, 'function_call': self.agent.function_call,
                'functions': self.agent.functions}

    def perform_api_call(self, api_function, params):
        max_retries = 3
        retry_delays = [10, 60, 300]  # Delays in seconds: 10s, 1min, 5min

        for attempt in range(max_retries):
            try:
                response = api_function(**params)
                time.sleep(0.15)  # Wait for 0.05 seconds before returning the response
                return response
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"Error during API call, retrying in {retry_delays[attempt]} seconds: {e}")
                    time.sleep(retry_delays[attempt])
                else:
                    print(f"Error during API call after {max_retries} attempts: {e}")
                    return None
        return None

    def filter_optional_params(self, params):
        default_values = {
            "frequency_penalty": 0.0,
            "function_call": "none",
            "functions": None,
            "logit_bias": None,
            "max_tokens": 50,
            "n": 1,
            "presence_penalty": 0.0,
            "stop": None,
            "stream": False,
            "temperature": 1.0,
            "top_p": 1.0,
        }

        return {k: v for k, v in params.items() if v != default_values.get(k, None)}

    def run_chat_agent(self, common_params):
        chat_params = self.prepare_messages()
        filtered_params = self.filter_optional_params({**common_params, **chat_params})
        completions = self.perform_api_call(openai.ChatCompletion.create, filtered_params)

        # Check if the completion conforms to the expected structure
        if isinstance(completions, dict) and 'choices' in completions:
            for choice in completions['choices']:
                if choice is None or \
                        not isinstance(choice, dict) or \
                        'message' not in choice or \
                        not isinstance(choice['message'], dict) or \
                        'content' not in choice['message']:
                    # If structure does not conform, return a fake completion
                    error_reason = 'API call failed for one of the completions or malformed completion'
                    fake_completion = {"content": f"Message is unavailable due to unforeseen events ({error_reason})."}
                    return [fake_completion]
            # If structure conforms, return the original completions
            return completions
        else:
            # If structure does not conform, return a fake completion
            error_reason = 'API call failed or unexpected completion format'
            fake_completion = {"content": f"Message is unavailable due to unforeseen events ({error_reason})."}
            return [fake_completion]

    def prepare_text_completion_params(self):
        prompt = self.agent.messages[-1] if self.agent.messages else None
        prompt_input_object = templates.TemplateManager.transform_into_prompt(prompt)
        return {
            'prompt': prompt_input_object.prompt_str,
            'echo': self.agent.echo,
            'logprobs': self.agent.logprobs,
            'suffix': self.agent.suffix
        }

    def prepare_embedding_params(self):
        text_to_embed = self.agent.messages
        embedding_object = templates.TemplateManager.transform_into_embedding(text_to_embed)
        return {'input': embedding_object.embedding_str, 'model': self.agent.model}

    def run_text_agent(self, common_params):
        completion_params = self.prepare_text_completion_params()
        filtered_params = self.filter_optional_params({**common_params, **completion_params})
        completion = self.perform_api_call(openai.Completion.create, filtered_params)

        if completion is None:
            error_reason = 'API call failed'
            fake_completion = [{"content": f"Message is unavailable due to unforeseen events ({error_reason})."}]
            return fake_completion
        else:
            return completion

    def run_embedding_agent(self):
        embedding_params = self.prepare_embedding_params()
        embedding_data = self.perform_api_call(openai.Embedding.create, embedding_params)

        if embedding_data is None:
            error_reason = 'API call failed'
            fake_completion = [{"content": f"Message is unavailable due to unforeseen events ({error_reason})."}]
            return fake_completion
        else:
            return embedding_data


class GPTAgent:
    """
    Class to handle GPT-based agents for both chat and completion tasks.
    """

    def __init__(self, model, **kwargs):
        self.model = model
        self.messages = None
        self.agent_name = None
        self.system_prompt = None
        self.echo = None
        self.frequency_penalty = None
        self.function_call = None
        self.functions = None
        self.logit_bias = None
        self.logprobs = None
        self.max_tokens = None
        self.n = None
        self.presence_penalty = None
        self.stop = None
        self.stream = None
        self.suffix = None
        self.temperature = None
        self.top_p = None
        self.user = None
        self.completion = None
        self.handler = GPTAgentHandler(self)

        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid keyword argument: {key}")

    def run_agent(self, token_pools_list=None):
        common_params = self.set_common_params()

        if self.model in ModelType.CHAT_MODELS:
            self.completion = self.handler.run_chat_agent(common_params)
        elif self.model in ModelType.TEXT_MODELS:
            self.completion = self.handler.run_text_agent(common_params)
        elif self.model in ModelType.EMBEDDING_MODELS:
            self.completion = self.handler.run_embedding_agent()

        if self.completion:
            self.add_token_pools(token_pools_list or [])

        return self.completion

    def update_agent(self, add_to_previous_chat_messages=False, **kwargs):
        """
        Updates the attributes of the GPTAgent instance.

        Keyword Arguments:
            append_messages (bool): If True, appends the new messages to the existing messages instead of replacing them.
            Any other attribute of the GPTAgent class can be updated.

        Example:
            # Update specific fields of the agent
            agent.update_agent(model="text-davinci-002", max_tokens=100, temperature=0.7)

            # Update multiple fields at once
            agent.update_agent(prompt="Tell me a story.", max_tokens=500, temperature=0.5, stop=["\n"])

            # Append new messages to existing messages
            agent.update_agent(append_messages=True, messages=[{"role":"system","content":"a new message here"}]
        """
        for key, value in kwargs.items():
            if key == "messages":
                if add_to_previous_chat_messages and hasattr(self, "messages"):
                    current_messages = getattr(self, "messages", [])
                    if isinstance(current_messages, list) and isinstance(value, list):
                        value = current_messages + value
                    elif isinstance(current_messages, list) and isinstance(value, dict):
                        value = current_messages + [value]
                    elif isinstance(current_messages, dict) and isinstance(value, list):
                        value = [current_messages] + value
                    elif isinstance(current_messages, dict) and isinstance(value, dict):
                        value = [current_messages, value]
                else:
                    if isinstance(value, dict):
                        value = [value]
            if hasattr(self, key):
                setattr(self, key, value)

    def set_common_params(self):
        """
        Sets the common parameters for both chat and completion agents.

        Returns:
            dict: A dictionary containing the common parameters.
        """
        return {
            'model': self.model,
            'frequency_penalty': self.frequency_penalty,
            'logit_bias': self.logit_bias,
            'max_tokens': self.max_tokens,
            'n': self.n,
            'presence_penalty': self.presence_penalty,
            'stop': self.stop,
            'stream': self.stream,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'user': self.user
        }

    # def prepare_messages(self):
    #     if isinstance(self.messages, list):
    #         messages_copy = self.messages.copy()
    #     elif isinstance(self.messages, dict):
    #         messages_copy = [self.messages]
    #     else:
    #         messages_copy = []
    #
    #     if self.system_prompt:
    #         if isinstance(self.system_prompt, str):
    #             system_prompt_dict = {'role': 'system', 'content': self.system_prompt}
    #             messages_copy.insert(0, system_prompt_dict)
    #         else:
    #             messages_copy.insert(0, self.system_prompt)
    #     else:
    #         default_system_prompt = {'role': 'system', 'content': 'you are a helpful AI assistant'}
    #         messages_copy.insert(0, default_system_prompt)
    #
    #     chat_message_object = templates.TemplateManager.transform_into_messages(messages_copy)
    #     prompt_string = " ".join([message_dict['content'] for message_dict in chat_message_object.messages])
    #     ModelManager.check_agent_token_limit(model=self.model, prompt=prompt_string, max_tokens=self.max_tokens)
    #
    #     return {'messages': chat_message_object.messages, 'function_call': self.function_call,
    #             'functions': self.functions}
    #
    # def perform_api_call(self, api_function, params):
    #     try:
    #         return api_function(**params)
    #     except Exception as e:
    #         print(f"Error during API call: {e}")
    #         return None
    #
    # # Main method for chat agent
    # def run_chat_agent(self, common_params):
    #     chat_params = self.prepare_messages()
    #     filtered_params = self.filter_optional_params({**common_params, **chat_params})
    #
    #     completions = self.perform_api_call(openai.ChatCompletion.create, filtered_params)
    #
    #     # Handling a list of dictionaries
    #     if isinstance(completions, list):
    #         processed_completions = []
    #
    #         for completion in completions:
    #             if completion is not None:
    #                 processed_completions.append(completion)
    #             else:
    #                 error_reason = 'API call failed for one of the completions'
    #                 fake_completion = {"content": f"Message is unavailable due to unforeseen events ({error_reason})."}
    #                 processed_completions.append(fake_completion)
    #
    #         return processed_completions
    #
    #     # Handling a single dictionary (for backward compatibility)
    #     elif completions is not None:
    #         return completions
    #
    #     # Handling failure
    #     else:
    #         error_reason = 'API call failed'
    #         fake_completion = [{"content": f"Message is unavailable due to unforeseen events ({error_reason})."}]
    #         return fake_completion
    #
    # def prepare_text_completion_params(self):
    #     prompt = self.messages[-1] if self.messages else None
    #     prompt_input_object = templates.TemplateManager.transform_into_prompt(prompt)
    #     return {
    #         'prompt': prompt_input_object.prompt_str,
    #         'echo': self.echo,
    #         'logprobs': self.logprobs,
    #         'suffix': self.suffix
    #     }
    #
    # # Method for preparing parameters for embedding
    # def prepare_embedding_params(self):
    #     text_to_embed = self.messages
    #     embedding_object = templates.TemplateManager.transform_into_embedding(text_to_embed)
    #     return {'input': embedding_object.embedding_str, 'model': self.model}
    #
    # def run_text_agent(self, common_params):
    #     completion_params = self.prepare_text_completion_params()
    #     filtered_params = self.filter_optional_params({**common_params, **completion_params})
    #     completion = self.perform_api_call(openai.Completion.create, filtered_params)
    #
    #     if completion is None:
    #         error_reason = 'API call failed'
    #         fake_completion = [{"content": f"Message is unavailable due to unforeseen events ({error_reason})."}]
    #         return fake_completion
    #     else:
    #         return completion
    #
    # # Main method for embedding
    # def run_embedding_agent(self):
    #     embedding_params = self.prepare_embedding_params()
    #     embedding_data = self.perform_api_call(openai.Embedding.create, embedding_params)
    #
    #     if embedding_data is None:
    #         error_reason = 'API call failed'
    #         fake_completion = [{"content": f"Message is unavailable due to unforeseen events ({error_reason})."}]
    #         return fake_completion
    #     else:
    #         return embedding_data

    def add_token_pools(self, token_pools_list):
        """
        Updates the token pools with the usage data from the completion.

        Parameters:
            completion (dict): The completion object returned by the GPT model.
            token_pools_list (list): List of token pool names to update.
        """

        token_pool_manager = token_pools.TokenPool()
        token_pool_manager.add_completion_to_pool(self.completion, token_pools_list)

    def get_finish_reason(self, index: Union[List[int], int] = 0):
        """
        Retrieves the finish reason(s) from the completion data.

        Parameters:
            index (Union[List[int], int]): Index or list of indices for which to retrieve the finish reason.

        Returns:
            str or List[str]: The finish reason(s) for the specified index/indices.

        Example Output:
            For chat: "length" OR "content_filter" OR "stop" OR "function_call"
            For text: "length" OR "content_filter" OR "stop"
        """
        if self.completion:
            if isinstance(index, list):
                return [self.completion["choices"][i]["finish_reason"] for i in index]
            return self.completion["choices"][index]["finish_reason"]
        return "No completion data available."

    def get_usage(self):
        """
        Retrieves the usage statistics from the completion data.

        Returns:
            dict: The usage statistics.

        Example Output:
            {'prompt_tokens': 9, 'completion_tokens': 12, 'total_tokens': 21}
        """
        if self.completion:
            return self.completion["usage"]
        return "No completion data available."

    def get_id(self):
        """
        Retrieves the ID from the completion object, created by openai.

        Returns:
            str: The ID of the completion.

        Example Output:
            "chatcmpl-123"
        """
        if self.completion:
            return self.completion["id"]
        return "No completion data available."

    def get_object(self):
        """
        Retrieves the object type from the completion data.

        Returns:
            str: The object type.

        Example Output:
            "chat.completion" OR "text_completion"
        """
        if self.completion:
            return self.completion["object"]
        return "No completion data available."

    def get_text(self, index: Union[List[int], int] = 0):
        """
        Retrieves the text content from the completion data.

        Parameters:
            index (Union[List[int], int]): Index or list of indices for which to retrieve the text content.

        Returns:
            str or List[str]: The text content for the specified index/indices.

        Example Output:
            For chat: "\n\nHello there, how may I assist you today?"
            For text: "\n\nThis is indeed a test"
        """
        if self.completion:
            if "message" in self.completion["choices"][0]:
                # This is a chat completion
                if isinstance(index, list):
                    return [self.completion["choices"][i]["message"]["content"] for i in index]
                return self.completion["choices"][index]["message"]["content"]
            else:
                # This is a text completion
                if isinstance(index, list):
                    return [self.completion["choices"][i]["text"] for i in index]
                return self.completion["choices"][index]["text"]
        return "No completion data available."

    def get_vector(self):
        """
        Retrieves the vector object from the completion data.

        Returns:
            list: The vector object if available, otherwise a message indicating it's not available.

        Example Output:
            [0.0023064255, -0.009327292, ...., -0.0028842222]
        """
        # Check if completion data is available and correctly structured
        if self.completion and 'data' in self.completion:
            for item in self.completion['data']:
                if item['object'] == 'embedding':
                    return item['embedding']
        return "No vector data available."

    def get_model_used(self):
        """
        Retrieves the model used for the completion.

        Returns:
            str: The model used if available, otherwise a message indicating it's not available.

        Example Output:
            "text-embedding-ada-002"
        """
        if self.completion and 'model' in self.completion:
            return self.completion['model']
        return "No model data available."


class GPTManager:
    """
    Manages the creation of GPT agents for various tasks (prompt, chat, embedding, etc.).
    """

    def __init__(self):
        """
        Initializes the GPTManager with necessary configurations.
        """
        config = config_retrieval.ConfigManager()
        openai.api_key = config.openai.api_key
        self.template_manager = templates.TemplateManager()
        self.model_manager = models.ModelManager()

    def create_agent(self,
                     model: ModelType,
                     messages: Union[List[dict], str, dict],
                     agent_name: str = None,
                     system_prompt: Union[dict, str] = None,
                     echo: Optional[bool] = False,
                     frequency_penalty: Optional[float] = 0.0,
                     function_call: Optional[Union[str, dict]] = "none",
                     functions: Optional[List] = None,
                     logit_bias: Optional[dict] = None,
                     logprobs: Optional[int] = None,
                     max_tokens: Optional[int] = 50,
                     n: Optional[int] = 1,
                     presence_penalty: Optional[float] = 0.0,
                     stop: Optional[Union[str, List[str]]] = None,
                     stream: Optional[bool] = False,
                     suffix: Optional[str] = None,
                     temperature: Optional[float] = 1.0,
                     top_p: Optional[float] = 1.0,
                     user: Optional[str] = "developper1") -> 'GPTAgent':
        """
        Creates a new GPTAgent instance.

        Parameters:
            model (str): Required. The ID of the model to use.

            messages (Union[List[str], str]): Required. A list of messages that make up the conversation history.

            echo (bool, optional): Whether to echo back the prompt. Defaults to False.

            frequency_penalty (float, optional): Between -2.0 and 2.0. Penalizes frequency of new tokens based on their existing frequency in the text. Defaults to 0.0.

            function_call (Union[str, dict], optional): Controls how the model calls functions. "none" means the model will not call a function. "auto" lets the model choose. Defaults to "none".

            functions (List, optional): A list of functions the model may generate JSON inputs for.

            logit_bias (dict, optional): Modifies the likelihood of specific tokens appearing in the completion. Maps tokens to a bias value between -100 and 100.

            logprobs (int, optional): Number of most likely token probabilities to return. Max value is 5.

            max_tokens (int, optional): Maximum number of tokens to generate in the chat completion. Defaults to 50.

            n (int, optional): Number of chat completion choices to generate. Defaults to 1.

            presence_penalty (float, optional): Between -2.0 and 2.0. Penalizes new tokens based on their presence in the text, encouraging new topics. Defaults to 0.0.

            stop (Union[str, List[str]], optional): Sequences where the API will stop generating tokens.

            stream (bool, optional): If set to True, partial message deltas will be sent. Defaults to False.

            suffix (str, optional): The suffix that comes after a completion of inserted text.

            temperature (float, optional): Sampling temperature between 0 and 2. Higher values make output more random. Defaults to 1.0.

            top_p (float, optional): Nucleus sampling parameter. The model considers tokens with top_p probability mass. Defaults to 1.0.

            user (str, optional): A unique identifier for the end-user to monitor and detect abuse. Defaults to "developper1".

        Returns:
            GPTAgent: A new instance of the GPTAgent class.
        """
        # Function body remains the same

        if not self.model_manager.check_agent_model(model):
            raise ValueError("Invalid model.")

        kwargs = {
            'agent_name': agent_name,
            'messages': messages,
            'system_prompt': system_prompt,
            'echo': echo,
            'frequency_penalty': frequency_penalty,
            'function_call': function_call,
            'functions': functions,
            'logit_bias': logit_bias,
            'logprobs': logprobs,
            'max_tokens': max_tokens,
            'n': n,
            'presence_penalty': presence_penalty,
            'stop': stop,
            'stream': stream,
            'suffix': suffix,
            'temperature': temperature,
            'top_p': top_p,
            'user': user
        }

        return GPTAgent(model=model, **kwargs)
