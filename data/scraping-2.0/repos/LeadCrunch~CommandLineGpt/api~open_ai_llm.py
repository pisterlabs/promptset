# -*- coding: utf-8 -*-

from __future__ import absolute_import, unicode_literals

import json

import openai

from config.main import openai_api_key

openai.api_key = openai_api_key


class OpenAiChatMessages:
    def __init__(self, messages):
        self.messages = [
            self.map_chat_message_to_openai_chat_message(message)
            for message in messages
        ]

    def map_chat_message_to_openai_chat_message(self, message):
        role = getattr(message, "role", "user")
        return {
            "role": self.map_role_to_openai_role(role),
            "content": getattr(message, "text", ""),
        }

    def map_role_to_openai_role(self, role):
        if role == "bot":
            return "assistant"
        return role

    @property
    def json(self):
        return self.messages


class OpenAiApiFunction:
    def __init__(self, name, description, parameters):
        self.name = name
        self.description = description
        self.parameters = parameters

    @property
    def json(self):
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class OpenAiChatCompletionResponse:
    choices = []

    class FunctionCall:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = json.loads(arguments)

    class Message:
        def __init__(self, message):
            self.role = getattr(message, "role", "user")
            self.content = getattr(message, "content", None)
            function_call = getattr(message, "function_call", None)
            self.function_call = (
                (
                    OpenAiChatCompletionResponse.FunctionCall(
                        function_call["name"], function_call["arguments"]
                    )
                )
                if function_call
                else None
            )

    class Choice:
        def __init__(self, choice):
            message = choice["message"]
            self.message = OpenAiChatCompletionResponse.Message(message)

    def __init__(self, chat_completion_response):
        self.choices = [
            self.Choice(choice) for choice in chat_completion_response["choices"]
        ]


class OpenAiLlmOptions:
    def __init__(self, model, temperature):
        self.model = model
        self.temperature = temperature


class OpenAiLlmApi:
    def __init__(self, options):
        self.options = options
        self.collected_response_text = ""
        self.status = "ready"

    def chat_completion(self, chat_messages=None, message=None):
        self.status = "processing"
        if message and not chat_messages:
            chat_messages = OpenAiChatMessages([{"role": "user", "content": message}])

        chat_completion = openai.ChatCompletion.create(
            model=self.options.model,
            temperature=self.options.temperature,
            messages=chat_messages.json,
        )
        self.status = "ready"
        return OpenAiChatCompletionResponse(chat_completion)

    def stream_chat_completion(self, chat_messages=None, message=None, callback=None):
        self.status = "processing"
        if message and not chat_messages:
            chat_messages = OpenAiChatMessages([{"role": "user", "content": message}])

        response = openai.ChatCompletion.create(
            messages=chat_messages.json,
            model=self.options.model,
            temperature=self.options.temperature,
            stream=True,
        )

        collected_events = []
        completion_text = ""

        for event in response:
            collected_events.append(event)
            if event["choices"][0]["finish_reason"] == "stop":
                self.status = "ready"
                break

            event_text = event["choices"][0]["delta"]["content"]
            completion_text += event_text
            self.collected_response_text += event_text
            
            if callback:
                callback(completion_text)

    def function_call(self, chat_messages, functions, function_call):
        chat_completion = openai.ChatCompletion.create(
            model=self.options.model,
            temperature=self.options.temperature,
            messages=chat_messages.json,
            functions=functions,
            function_call=function_call,
        )
        return (
            OpenAiChatCompletionResponse(chat_completion)
            .choices[0]
            .message.function_call.arguments
        )
