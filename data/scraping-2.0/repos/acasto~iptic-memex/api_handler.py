import os
import openai
import tiktoken
from abc import ABC, abstractmethod

class APIHandler(ABC):
    """
    Abstract class for API handlers
    """
    @abstractmethod
    def complete(self, prompt):
        pass

    @abstractmethod
    def stream_complete(self, prompt):
        pass

    @abstractmethod
    def chat(self, message):
        pass

    @abstractmethod
    def stream_chat(self, message):
        pass


# noinspection PyTypeChecker
class OpenAIHandler(APIHandler):
    """
    OpenAI API handler
    """
    def __init__(self, conf):
        self.conf = conf
        if 'api_key' in conf:
            self.api_key = conf['api_key']
        else:
            self.api_key = os.environ['OPENAI_API_KEY']
        openai.api_key = self.api_key

    def complete(self, prompt):
        """
        Creates a completion request to the OpenAI API
        :param prompt: the prompt to complete from an interaction handler
        :return: response (str)
        """
        try:
            response = openai.Completion.create(
                request_timeout=120,
                model=self.conf['model'],
                prompt=prompt,
                temperature=float(self.conf['temperature']),
                max_tokens=int(self.conf['max_tokens']),
                stream=bool(self.conf['stream']),
            )
            # if in stream mode chain the generator
            if self.conf['stream']:
                return response
            else:
                return response.choices[0].text.strip()
        except openai.error.InvalidRequestError as e:
            # Handle token limit errors
            if 'maximum context length ' in str(e):
                return "Error: Token count exceeds the limit."
        except openai.error.APIConnectionError:
            # Handle timeout errors
            return "Error: Connection timeout. Please try again later."
        except Exception as e:
            # Handle other exceptions
            return f"An unexpected error occurred: {str(e)}"

    def stream_complete(self, prompt):
        """
        Use generator chaining to keep the response provider-agnostic
        :param prompt:
        :return:
        """
        response =  self.complete(prompt)
        if type(response) == str:
            yield response
        else:
            for event in response:
                yield event['choices'][0]['text']

    def chat(self, messages):
        """
        Creates a chat completion request to the OpenAI API
        :param messages: the message to complete from an interaction handler
        :return: response (str)
        """
        try:
            response = openai.ChatCompletion.create(
                request_timeout=120,
                model=self.conf['model'],
                messages=messages,
                temperature=float(self.conf['temperature']),
                max_tokens=int(self.conf['max_tokens']),
                stream=bool(self.conf['stream']),
            )
            # if in stream mode chain the generator
            if self.conf['stream']:
                return response
            else:
                return response['choices'][0]['message']['content']
        except openai.error.InvalidRequestError as e:
            # Handle token limit errors
            if 'maximum context length ' in str(e):
                return "Error: Token count exceeds the limit."
        except openai.error.APIConnectionError:
            # Handle timeout errors
            return "Error: Connection timeout. Please try again later."
        except Exception as e:
            # Handle other exceptions
            return f"An unexpected error occurred: {str(e)}"

    def stream_chat(self, messages):
        """
        Use generator chaining to keep the response provider-agnostic
        :param messages:
        :return:
        """
        response = self.chat(messages)
        if type(response) == str:
            yield response
        else:
            for event in response:
                if 'content' in event['choices'][0]['delta']:
                    yield event['choices'][0]['delta']['content']

    # See: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    def count_tokens(self, messages, model="gpt-3.5-turbo-0613"):
        """Returns the number of tokens used"""
        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")

        if model in {
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k",
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = 4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )

        num_tokens = 0
        # if we're given a str for a completion prompt, convert it to a list
        if not isinstance(messages, list):
            messages = [messages]
        # process list of messages
        for message in messages:
            num_tokens += tokens_per_message
            for key, value in message.items():
                num_tokens += len(encoding.encode(value))
                if key == "role":
                    num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
        return num_tokens


class OpenRouterHandler(APIHandler):
    """
    OpenAI API handler
    """
    def __init__(self, conf):
        self.conf = conf
        if 'api_key' in conf:
            self.api_key = conf['api_key']
        else:
            self.api_key = os.environ['OPENROUTER_API_KEY']
        openai.api_key = self.api_key
        openai.api_base = "https://openrouter.ai/api/v1"

    def complete(self, prompt):
        """
        Creates a completion request to the OpenAI API
        :param prompt: the prompt to complete from an interaction handler
        :return: response (str)
        """
        try:
            response = openai.Completion.create(
                request_timeout=120,
                model=self.conf['model'],
                prompt=prompt,
                temperature=float(self.conf['temperature']),
                max_tokens=int(self.conf['max_tokens']),
                stream=bool(self.conf['stream']),
            )
            # if in stream mode chain the generator
            if self.conf['stream']:
                return response
            else:
                return response.choices[0].text.strip()
        except openai.error.InvalidRequestError as e:
            # Handle token limit errors
            if 'maximum context length ' in str(e):
                return "Error: Token count exceeds the limit."
        except openai.error.APIConnectionError:
            # Handle timeout errors
            return "Error: Connection timeout. Please try again later."
        except Exception as e:
            # Handle other exceptions
            return f"An unexpected error occurred: {str(e)}"

    def stream_complete(self, prompt):
        """
        Use generator chaining to keep the response provider-agnostic
        :param prompt:
        :return:
        """
        response =  self.complete(prompt)
        if type(response) == str:
            yield response
        else:
            for event in response:
                yield event['choices'][0]['text']

    def chat(self, messages):
        """
        Creates a chat completion request to the OpenAI API
        :param messages: the message to complete from an interaction handler
        :return: response (str)
        """
        try:
            response = openai.ChatCompletion.create(
                request_timeout=120,
                model=self.conf['model'],
                messages=messages,
                temperature=float(self.conf['temperature']),
                max_tokens=int(self.conf['max_tokens']),
                stream=bool(self.conf['stream']),
                headers= { "HTTP-Referer": "https://iptic.com",
                           "X-Title": "iptic-memex" },
            )
            # if in stream mode chain the generator
            if self.conf['stream']:
                return response
            else:
                return response['choices'][0]['message']['content']
        except openai.error.InvalidRequestError as e:
            # Handle token limit errors
            if 'maximum context length ' in str(e):
                return "Error: Token count exceeds the limit."
        except openai.error.APIConnectionError:
            # Handle timeout errors
            return "Error: Connection timeout. Please try again later."
        except Exception as e:
            # Handle other exceptions
            return f"An unexpected error occurred: {str(e)}"

    def stream_chat(self, messages):
        """
        Use generator chaining to keep the response provider-agnostic
        :param messages:
        :return:
        """
        response = self.chat(messages)
        if type(response) == str:
            yield response
        else:
            for event in response:
                if 'content' in event['choices'][0]['delta']:
                    yield event['choices'][0]['delta']['content']
