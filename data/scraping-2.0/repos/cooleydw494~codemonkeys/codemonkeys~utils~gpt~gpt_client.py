import json
import time
from typing import List, Optional, Any

import openai
import tiktoken
from json_repair import repair_json
from tiktoken import Encoding

from codemonkeys.defs import TOKEN_UNCERTAINTY_BUFFER, nl
from codemonkeys.entities.func import Func
from codemonkeys.types import OStr
from codemonkeys.utils.config.monkey_validations import validate_model, validate_temp
from codemonkeys.utils.imports.env import Env
from codemonkeys.utils.misc.handle_exception import handle_exception
from codemonkeys.utils.misc.log import Log
from codemonkeys.utils.monk.theme_functions import print_t


class GptClient:
    """A helper class to interact with GPT based models."""

    def __init__(self, model_name: str, temperature: float = 1.0, max_tokens: int = 8000, max_response_tokens: int = 4096):
        """
        Initialize a GptClient instance for a specific model.
        
        :param str model_name: The model's name.
        :param float temperature: The generation temperature. Defaults to 1.0.
        :param int max_tokens: Maximum tokens limit. Defaults to 8000.
        """

        env = Env.get()
        self._openai = openai.OpenAI(api_key=env.OPENAI_API_KEY)

        self.model: str = validate_model(model_name)
        self.max_tokens: int = max_tokens - TOKEN_UNCERTAINTY_BUFFER
        self.max_response_tokens: int = max_response_tokens
        self.temperature: float = validate_temp(temperature)
        self.encoding: Encoding = tiktoken.encoding_for_model(self.model)

    def generate(self, prompt: str, funcs: Optional[List[Func]] = None, enforce_func: OStr = None,
                 retry_delay: int = 60) -> Any:
        """
        Generate a GPT model response from a given prompt.

        Utilizes the specified GPT model and settings to create a response based on the input prompt,
        optionally including function calling with predefined funcs. Handles rate limit errors by retrying
        after a specified delay, doubling the delay with each retry.

        :param enforce_func: The name of the function to enforce. Defaults to None.
        :param funcs: list of Func classes to use for function calling.
        :param str prompt: The text input for the model.
        :param int retry_delay: The delay in seconds to wait before rate limit retry. Defaults to 60.
        :return: The generated response.
        """

        try:
            if funcs:
                return self._generate_with_funcs(prompt, funcs, enforce_func)
            else:
                return self._generate(prompt)
        except openai.RateLimitError as e:
            print_t(f"Rate limit error, trying again in {retry_delay}s", 'warning')
            handle_exception(e, always_continue=True)
            time.sleep(retry_delay)
            return self.generate(prompt, funcs, enforce_func, retry_delay * 2)

        # Commented out because timeouts sometimes recur many times but only for specific prompts (best to skip)
        #
        # except openai.Timeout as e:
        #     print_t(f"Timeout error, trying again in {rate_limit_delay}s: {e}", 'warning')
        #     import time
        #     time.sleep(rate_limit_delay)
        #     return self.generate(_prompt, rate_limit_delay*2)

        except openai.OpenAIError as e:
            print_t(f"OpenAIError generating GPT response.", 'error')
            handle_exception(e, always_continue=True)
        except BaseException as e:
            print_t(f"GPT call failed.", 'error')
            handle_exception(e, always_continue=True)
        return None

    def _generate(self, prompt: str) -> str:
        """
        Generate a GPT response from a given prompt.
        :param prompt: The text input for the model.
        :return: The generated response.
        """

        max_tokens = self.max_tokens - self.count_tokens(prompt)
        max_tokens = min(max_tokens, self.max_response_tokens)
        print()
        print_t(f"Generating with {max_tokens}/{self.max_tokens} tokens remaining for response", 'loading')

        response = self._openai.chat.completions.create(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=max_tokens,
            temperature=self.temperature
        )

        return response.choices[0].message.content.strip()

    def _generate_with_funcs(self, prompt: str, funcs: Optional[List[Func]] = None, enforce_func: OStr = None) -> Any:
        """
        Generate a GPT model response from a given prompt, using the given Funcs (function calling)
        :param prompt: The text input for the model.
        :param funcs: list of Func classes to use for function calling.
        :param enforce_func: The name of the function to enforce. Defaults to None / "auto".
        :return: The result of the Func call (can be anything).
        """

        tools = [func.data() for func in funcs]
        tool_choice = {'type': 'function', 'function': {'name': enforce_func}} if enforce_func else 'auto'

        max_tokens = self.max_tokens - self.count_tokens(prompt) - self.count_tokens(json.dumps(tools))
        max_tokens = min(max_tokens, self.max_response_tokens)
        print()
        print_t(f"Generating with {max_tokens}/{self.max_tokens} tokens remaining for response", 'loading')

        response = self._openai.chat.completions.create(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=max_tokens,
            temperature=self.temperature,
            tools=tools,
            tool_choice=tool_choice
        )

        function_response = response.choices[0].message.tool_calls[0].function
        (name, args) = (function_response.name, function_response.arguments)
        available_funcs = {func.name: func for func in funcs}
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            print_t(f'sanitizing args due to decoding error', 'special', verbose=True)
            Log.warning(f'function call ({name}) produced invalid raw JSON args:{nl}{repr(args)}')
            args = json.loads(repair_json(args))
            Log.warning(f'function call ({name}) sanitized JSON args:{nl}{repr(args)}')

        func = available_funcs[name]

        return func.call(args)

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in a given text.

        :param str text: The input text.
        :return: The number of tokens in the text.
        """
        return len(self._tokenize(text))

    def _tokenize(self, text: str) -> List[int]:
        """
        Tokenize a given text into a list of tokens.

        :param str text: The input text.
        :return: The list of tokens.
        """
        return self.encoding.encode(text)
