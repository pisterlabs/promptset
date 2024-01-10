#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/2/10 10:31
# @Author  : Jack
# @File    : official.py
# @Software: PyCharm

"""
A simple wrapper for the official ChatGPT API
"""
import json
import openai
import tiktoken

DEFAULT_ENGINE = "text-davinci-003"

ENCODER = tiktoken.get_encoding("gpt2")


def get_max_tokens(prompt: str) -> int:
    """
    Get the max tokens for a prompt
    """
    return 4000 - len(ENCODER.encode(prompt))


def remove_suffix(input_string, suffix):
    """
    Remove suffix from string (Support for Python 3.8)
    """
    if suffix and input_string.endswith(suffix):
        return input_string[: -len(suffix)]
    return input_string


class Prompt:
    """
    Prompt with methods to construct prompt
    """

    def __init__(self, custom_base_prompt: str, buffer: int = None) -> None:
        """
        Initialize prompt with base prompt
        :param buffer:
        """
        init_base_prompt = "You are ChatGPT, a large language model trained by OpenAI. Respond conversationally."
        self.base_prompt = custom_base_prompt or init_base_prompt
        # Track chat history
        self.chat_history: list = []
        self.buffer = buffer

    def add_to_chat_history(self, chat: str) -> None:
        """
        Add chat to chat history for next prompt
        :param chat: previous chat
        """
        self.chat_history.append(chat)

    def add_to_history(self, user_request: str, response: str, user: str = "User") -> None:
        """
        Add request/response to chat history for next prompt
        """
        self.add_to_chat_history(
            user
            + ": "
            + user_request
            + "\n\n\n"
            + "ChatGPT: "
            + response
            + "\n",
        )

    def history(self, custom_history: list = None) -> str:
        """
        Return chat history
        """
        return "\n".join(custom_history or self.chat_history)

    def construct_prompt(self, new_prompt: str, custom_history: list = None, user: str = "User") -> str:
        """
        Construct prompt based on chat history and request
        """
        prompt = f'{self.base_prompt}{self.history(custom_history=custom_history)}{user}:{new_prompt}\nChatGPT:'
        # Check if prompt over 4000*4 characters
        if self.buffer is not None:
            max_tokens = 4000 - self.buffer
        else:
            max_tokens = 3200
        if len(ENCODER.encode(prompt)) > max_tokens:
            # Remove the oldest chat
            if len(self.chat_history) == 0:
                return prompt
            self.chat_history.pop(0)
            # Construct prompt again
            prompt = self.construct_prompt(new_prompt, custom_history, user)
        return prompt


class Conversation:
    """
    For handling multiple conversations
    """

    def __init__(self) -> None:
        self.conversations = {}

    def add_conversation(self, key: str, history: list) -> None:
        """
        Adds a history list to the conversations dict with the id as the key
        """
        self.conversations[key] = history

    def get_conversation(self, key: str) -> list:
        """
        Retrieves the history list from the conversations dict with the id as the key
        """
        return self.conversations[key]

    def remove_conversation(self, key: str) -> None:
        """
        Removes the history list from the conversations dict with the id as the key
        """
        del self.conversations[key]

    def __str__(self) -> str:
        """
        Creates a JSON string of the conversations
        """
        return json.dumps(self.conversations)

    def save(self, file: str) -> None:
        """
        Saves the conversations to a JSON file
        """
        with open(file, "w", encoding="utf-8") as f:
            f.write(str(self))

    def load(self, file: str) -> None:
        """
        Loads the conversations from a JSON file
        """
        with open(file, encoding="utf-8") as f:
            self.conversations = json.loads(f.read())


class Chatbot:
    """
    Official Openai API
    """
    def __init__(
            self,
            api_key: str = None,
            org_id: str = None,
            buffer: int = None,
            engine: str = None,
            proxy: str = None,
            custom_base_prompt: str = None,
    ) -> None:
        """
        Initialize Chatbot with API key (from https://platform.openai.com/account/api-keys)
        """
        openai.api_key = api_key
        openai.organization = org_id
        openai.proxy = proxy
        self.conversations = Conversation()
        self.prompt = Prompt(custom_base_prompt=custom_base_prompt, buffer=buffer)
        self.engine = engine or DEFAULT_ENGINE

    def _get_completion(
            self,
            prompt: str,
            temperature: float = 0.5,
            stream: bool = False,
    ):
        """
        Get the openai API response
        """
        return openai.Completion.create(
            engine=self.engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=get_max_tokens(prompt),
            stop=["\n\n\n"],
            stream=stream,
        )

    def _process_completion(
            self,
            user_request: str,
            completion: dict,
            conversation_id: str = None,
            user: str = "User",
    ) -> str:
        if completion.get("choices") is None:
            raise Exception("Openai API returned no choices")
        if len(completion["choices"]) == 0:
            raise Exception("Openai API returned no choices")
        if completion["choices"][0].get("text") is None:
            raise Exception("Openai API returned no text")
        # Add to chat history
        self.prompt.add_to_history(
            user_request,
            completion["choices"][0]["text"],
            user=user,
        )
        if conversation_id is not None:
            self.save_conversation(conversation_id)
        return completion["choices"][0]["text"]

    def _process_completion_stream(
            self,
            user_request: str,
            completion: dict,
            conversation_id: str = None,
            user: str = "User",
    ) -> str:
        full_response = ""
        for response in completion:
            if response.get("choices") is None:
                raise Exception("ChatGPT API returned no choices")
            if len(response["choices"]) == 0:
                raise Exception("ChatGPT API returned no choices")
            if response["choices"][0].get("finish_details") is not None:
                break
            if response["choices"][0].get("text") is None:
                raise Exception("ChatGPT API returned no text")
            yield response["choices"][0]["text"]
            full_response += response["choices"][0]["text"]

        # Add to chat history
        self.prompt.add_to_history(user_request, full_response, user)
        if conversation_id is not None:
            self.save_conversation(conversation_id)

    def ask(
            self,
            user_request: str,
            temperature: float = 0.5,
            conversation_id: str = None,
            user: str = "User",
    ) -> str:
        """
        Send a request to ChatGPT and return the response
        """
        if conversation_id is not None:
            self.load_conversation(conversation_id)
        completion = self._get_completion(
            self.prompt.construct_prompt(user_request, user=user),
            temperature,
        )
        return self._process_completion(user_request, completion, user=user)

    def ask_stream(
            self,
            user_request: str,
            temperature: float = 0.5,
            conversation_id: str = None,
            user: str = "User",
    ) -> str:
        """
        Send a request to ChatGPT and yield the response
        """
        if conversation_id is not None:
            self.load_conversation(conversation_id)
        prompt = self.prompt.construct_prompt(user_request, user=user)
        return self._process_completion_stream(
            user_request=user_request,
            completion=self._get_completion(prompt, temperature, stream=True),
            user=user,
        )

    def make_conversation(self, conversation_id: str) -> None:
        """
        Make a conversation
        """
        self.conversations.add_conversation(conversation_id, [])

    def rollback(self, num: int) -> None:
        """
        Rollback chat history num times
        """
        for _ in range(num):
            self.prompt.chat_history.pop()

    def reset(self) -> None:
        """
        Reset chat history
        """
        self.prompt.chat_history = []

    def load_conversation(self, conversation_id) -> None:
        """
        Load a conversation from the conversation history
        """
        if conversation_id not in self.conversations.conversations:
            # Create a new conversation
            self.make_conversation(conversation_id)
        self.prompt.chat_history = self.conversations.get_conversation(conversation_id)

    def save_conversation(self, conversation_id) -> None:
        """
        Save a conversation to the conversation history
        """
        self.conversations.add_conversation(conversation_id, self.prompt.chat_history)
