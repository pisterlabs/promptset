from typing import Any, List, Dict
import openai
from functools import partial, wraps
import time
import asyncio
import os
from typing import Any, Union
from dotenv import load_dotenv
from .utils import sanitize_text


try:
    import openai
except ImportError:
    raise ImportError(
        "Please install the openai package with `pip install openai`"
    )

load_dotenv(".env")


class Completion:
    def __init__(self, temperature=0.7, max_tokens=1000, stream=False, model="text-davinci-003", api_key=os.getenv("OPENAI_API_KEY")):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        self.model = model
        self.api_key = api_key

    def __call__(self, text):
        """
        Process the user message and return the assistant's response.
        """

        openai.api_key = self.api_key

        raw_response = openai.Completion.create(
            model=self.model,
            prompt=text,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if self.stream:
            return raw_response
        elif len(raw_response['choices']) > 1:
            res_dicts = []
            for idx, choice in enumerate(raw_response['choices']):
                response = choice['text'].strip()
                tokens = raw_response['usage']['total_tokens']
                res_dict = {"response": response, "model": self.model,
                            "temperature": self.temperature, "tokens": tokens}
                res_dicts.append(res_dict)
            return res_dicts
        else:
            response = raw_response['choices'][0]['text'].strip()
            tokens = raw_response['usage']['total_tokens']
            return {"response": response, "model": self.model, "temperature": self.temperature, "tokens": tokens}

class ChatServer:
    """
    A class to interact with the OpenAI Chat API.
    """

    def __init__(self, api_key=os.getenv("OPENAI_API_KEY")):
        """
        Initialize the Chat class with the given parameters.
        """
        self.api_key = api_key
        openai.api_key = self.api_key

    def __call__(self, messages, temperature=0, model='gpt-3.5-turbo-16k', max_tokens=2048, stream=True):
        raw_response = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )
        return raw_response


class Chat:
    """
    A class to interact with the OpenAI Chat API.
    """

    def __init__(self, temperature=0.7, system_message="You are a helpful assistant.", messages=None, model='gpt-3.5-turbo', max_tokens=2000, stream=False, api_key=os.getenv("OPENAI_API_KEY")):
        """
        Initialize the Chat class with the given parameters.
        """
        self.messages = []
        self.messages.append({"role": "system", "content": system_message})
        if messages is not None:
            self.messages += [{"role": "user", "content": message}
                              for message in messages]
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_key = api_key
        self.stream = stream

    def __call__(self, user_message: str):
        """
        Process the user message and return the assistant's response.
        """
        openai.api_key = self.api_key

        user_message = {"role": "user", "content": user_message}
        self.messages.append(user_message)
        raw_response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=self.stream,
        )

        if self.stream is True:
            return raw_response
        
        text = raw_response['choices'][0]['message']['content'].strip()
        tokens = raw_response['usage']['total_tokens']
        res_message = {"role": "assistant", "content": text}
        self.messages.append(res_message)
        res_dict = {"response": text, "messages": self.messages,
                    "model": self.model, "temperature": self.temperature, "tokens": tokens}
        return res_dict


# ! ASYNC CHAT --------------------------------------------------------


def async_wrap(func):
    @wraps(func)
    async def run(*args, loop=None, executor=None, **kwargs):
        if loop is None:
            loop = asyncio.get_event_loop()
        pfunc = partial(func, *args, **kwargs)
        return await loop.run_in_executor(executor, pfunc)
    return run


class ChatAsync:
    """
    A class to handle asynchronous chat-based interactions with OpenAI's chat models.
    """

    def __init__(self, api_key=os.getenv("OPENAI_API_KEY"), model="gpt-3.5-turbo"):
        """
        Initialize the ChatAssistant with the provided API key and model.

        :param api_key: str, OpenAI API key
        :param model: str, the name of the OpenAI model to use (default: "gpt-3.5-turbo")
        """
        openai.api_key = api_key
        self.model = model

    def chat_response(self, temperature, message, max_tokens, system_message):
        """
        Generate a chat response using the OpenAI API.

        :param temperature: float, sampling temperature for the model (0 to 1)
        :param message: str, the user's message to the assistant
        :param max_tokens: int, maximum number of tokens in the response
        :param system_message: str, the initial system message to set the context
        :return: dict, a dictionary containing the response, messages, model, temperature, and tokens
        """
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": message}
        ]
        raw_response = openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        text = raw_response['choices'][0]['message']['content'].strip()
        tokens = raw_response['usage']['total_tokens']
        res_message = {"role": "assistant", "content": text}
        messages.append(res_message)
        res_dict = {"response": text, "messages": messages,
                    "model": self.model, "temperature": temperature, "tokens": tokens}
        return res_dict

    chat_response_wrapped = async_wrap(chat_response)

    async def async_chat_response(self, temperature, message, max_tokens, system_message, response_list, messages_list):
        """
        Asynchronously generate a chat response and update the response list.

        :param temperature: float, sampling temperature for the model (0 to 1)
        :param message: str, the user's message to the assistant
        :param max_tokens: int, maximum number of tokens in the response
        :param system_message: str, the initial system message to set the context
        :param response_list: list, a list to store the generated responses
        :param messages_list: list, a list of user messages
        """
        start_time = time.perf_counter()
        response = await self.chat_response_wrapped(temperature, message, max_tokens, system_message)
        elapsed = time.perf_counter() - start_time

        index = messages_list.index(message) + 1
        length = len(messages_list)
        print(f"Response {index} of {length} complete.")
        print(f"Response time: {elapsed:0.2f} seconds.")
        response_list.append(response)

    async def run_chat_async(self, messages_list, response_list, max_tokens=1000, temperature=0.7, system_message="You are a helpful assistant."):
        """
        Asynchronously generate chat responses for a list of messages.

        :param messages_list: list, a list of user messages
        :param response_list: list, a list to store the generated responses
        :param max_tokens: int, maximum number of tokens in the response (default: 1000)
        :param temperature: float, sampling temperature for the model (0 to 1, default: 0.7)
        :param system_message: str, the initial system message to set the context (default: "You are a helpful assistant.")
        """
        await asyncio.gather(*(self.async_chat_response(temperature, message, max_tokens, system_message, response_list, messages_list) for message in messages_list))

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        asyncio.run(self.run_chat_async(*args, **kwargs))

# ! EMBEDDINGS --------------------------------------------------------


def embed_ada(text: str):
    """
    Embed a text string using the ADA model.
    """
    if not isinstance(text, str):
        raise TypeError(
            "Text must be a string. Use embed_ada_list() to embed a list of strings.")

    sanitized_text = sanitize_text(text).replace("\n", " ").strip()
    if sanitized_text == "":
        raise ValueError("Empty text passed to embed_text()")

    # Embed the text
    response = openai.Embedding.create(
        input=sanitized_text,
        model="text-embedding-ada-002",
    )
    embedding = response["data"][0]["embedding"]
    return embedding


def embed_ada_list(text_list: List):
    if not isinstance(text_list, list):
        raise TypeError(
            "Text must be a list. Use embed_ada() to embed a single string.")
    sanitized_list = [sanitize_text(t).replace(
        "\n", " ").strip() for t in text_list if t != ""]
    if len(sanitized_list) == 0:
        raise ValueError("Empty list passed to embed_text()")
    # Embed the text
    response = openai.Embedding.create(
        input=sanitized_list,
        model="text-embedding-ada-002",
    )
    embeddings = [item["embedding"] for item in response["data"]]
    return embeddings