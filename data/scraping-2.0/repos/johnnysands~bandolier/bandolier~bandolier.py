"""Experiment with using OpenAI chat functions."""

from box import Box
import inspect
import openai
import json
import tiktoken


def completion(model, messages, functions=None):
    request = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
    }
    if functions:
        request["functions"] = functions

    response = openai.ChatCompletion.create(**request)
    return response["choices"][0]


class Bandolier:
    def __init__(
        self, completion_fn=completion, model="gpt-3.5-turbo", max_tokens=3000
    ):
        self.functions = {}
        self.function_metadata = []
        self.completion_fn = completion_fn
        self.model = model
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model(model)

    def add_function(self, function):
        name = function.__name__
        description = function.__doc__ if hasattr(function, "__doc__") else ""
        properties = (
            function.__properties__ if hasattr(function, "__properties__") else {}
        )

        # Get the list of arguments from the function signature
        signature = inspect.signature(function)
        function_args = set(signature.parameters.keys())

        properties_args = set(properties.keys())
        if function_args != properties_args:
            raise ValueError(f"Arguments for function {name} do not match the schema.")

        required = []
        for param_name, param in signature.parameters.items():
            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        metadata = {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties},
            "required": required,
        }
        self.functions[name] = function
        self.function_metadata.append(metadata)

    def call(self, function_name, arguments):
        arguments = json.loads(arguments)
        function = self.functions[function_name]
        return Box(
            {
                "role": "function",
                "name": function_name,
                "content": json.dumps(function(**arguments)),
            }
        )

    def run(self, conversation):
        """Run the chatbot using the supplied conversation.
        returns a list of all new messages."""

        response = self.completion_fn(
            self.model, conversation.messages, self.function_metadata
        )
        conversation.add_message(response.message)

        # maintain conversation at a usable size.
        conversation.trim_messages(self.encoding, self.max_tokens)

        messages = [response.message]
        while response.finish_reason == "function_call":
            # call function and store message
            fn_message = self.call(
                response.message.function_call.name,
                response.message.function_call.arguments,
            )
            conversation.add_message(fn_message)
            messages.append(fn_message)

            # return response from function
            response = self.completion_fn(
                self.model, conversation.messages, self.function_metadata
            )
            conversation.add_message(response.message)
            messages.append(response.message)

        if response.finish_reason != "stop":
            raise Exception(f"Unexpected finish reason: {response.finish_reason}")
        return messages
