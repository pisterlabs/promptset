from copy import deepcopy
from dataclasses import dataclass, field
from typing import Optional

from tiktoken import encoding_for_model, get_encoding

from openai_token_counter.format import format_function_definitions

from .models import OpenAIFunction, OpenAIMessage, OpenAIRequest


@dataclass
class TokenCounter:
    """Token counter class.

    Attributes:
        model (Optional[str]): The model to use for token counting.
    """

    model: Optional[str] = field(default=None)

    def estimate_token_count(self, request: OpenAIRequest) -> int:
        """Estimate the number of tokens a prompt will use.

        Args:
            request (OpenAIRequest): The request to estimate the token count for.

        Returns:
            int: An estimate for the number of tokens the prompt will use.
        """
        messages = request.messages
        functions = request.functions
        function_call = request.function_call

        padded_system = False
        tokens = 0

        for message in messages:
            message_copy = deepcopy(message)
            if message_copy.role == "system" and functions and not padded_system:
                if message_copy.content:
                    message_copy.content += "\n"
                padded_system = True
            tokens += self.estimate_tokens_in_messages(message_copy)

        # Each completion (vs message) seems to carry a 3-token overhead
        tokens += 3

        # If there are functions, add the function definitions as they count towards token usage
        if functions:
            tokens += self.estimate_tokens_in_functions(functions)

        # If there's a system message _and_ functions are present, subtract four tokens
        if functions and any(message.role == "system" for message in messages):
            tokens -= 4

        # If function_call is 'none', add one token.
        # If it's a OpenAIFunctionCall object, add 4 + the number of tokens in the function name.
        # If it's undefined or 'auto', don't add anything.
        if function_call and function_call != "auto":
            if function_call == "none":
                tokens += 1

            elif isinstance(function_call, dict) and "name" in function_call:
                tokens += self.string_tokens(function_call["name"]) + 4

        return tokens

    def string_tokens(self, string: str) -> int:
        """Get the token count for a string.

        Args:
            string (str): The string to count.

        Returns:
            int: The token count.
        """
        if self.model:
            encoding = encoding_for_model(self.model)
        else:
            encoding = get_encoding("cl100k_base")

        return len(encoding.encode(string))

    def estimate_tokens_in_messages(self, message: OpenAIMessage) -> int:
        """Estimate token count for a single message.

        Args:
            message (OpenAIMessage): The message to estimate the token count for.

        Returns:
            int: The estimated token count.
        """
        tokens = 0

        if message.role:
            tokens += self.string_tokens(message.role)

        if message.content:
            tokens += self.string_tokens(message.content)

        if message.name:
            tokens += self.string_tokens(message.name) + 1  # +1 for the name

        if message.function_call:
            if message.function_call.name:
                tokens += self.string_tokens(message.function_call.name)

            if message.function_call.arguments:
                tokens += self.string_tokens(message.function_call.arguments)

            tokens += 3  # Additional tokens for function call

        tokens += 3  # Add three per message

        if message.role == "function":
            tokens -= 2  # Subtract 2 if role is "function"

        return tokens

    def estimate_tokens_in_functions(self, function: list[OpenAIFunction]) -> int:
        """Estimate token count for the functions.

        We take here a list of functions and not one function because we use the format_function_definitions function
        which has to takes a list of functions as input, since the formatting is done for all functions.

        Args:
            function (list[OpenAIFunction]): The functions to estimate the token count for.

        Returns:
            int: The estimated token count.
        """
        prompt_definition = format_function_definitions(function)
        tokens = self.string_tokens(prompt_definition)
        tokens += 9  # Additional tokens for function definition
        return tokens
