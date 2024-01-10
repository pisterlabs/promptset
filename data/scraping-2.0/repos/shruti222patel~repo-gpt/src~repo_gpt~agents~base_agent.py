import inspect
import json
import logging
from abc import ABC, abstractmethod

import openai
import tiktoken
from tenacity import (  # for exponential backoff
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

from repo_gpt.agents.simple_memory_store import MemoryStore

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    GPT_MODEL = "gpt-3.5-turbo-0613"  # gpt-4-0613

    def __init__(
        self,
        user_task,
        terminating_function_call_name,
        system_prompt,
        threshold=10,
        debug=False,
    ):
        self.terminating_function_call_name = terminating_function_call_name
        self.functions = self._initialize_functions()
        self.memory_store = MemoryStore(system_prompt, user_task, self.functions)
        self.user_task = user_task
        self.system_prompt = system_prompt
        self.threshold = threshold
        self.debug = debug

    @abstractmethod
    def _initialize_functions(self):
        """
        Must be implemented by subclasses to initialize function-related attributes.
        """
        pass

    def _parse_arguments(self, function_call):
        return json.loads(function_call["arguments"])

    def _append_message(self, message):
        self.memory_store.add_message(message)

    def compress_messages(self):
        self.memory_store.compress_messages()

    def execute_function_call(self, message):
        function_name = message["function_call"]["name"]
        args = self._parse_arguments(message["function_call"])

        func = getattr(self, function_name, None)
        if not func:
            return f"Error: function {function_name} does not exist"

        # Filter out args to only pass those that the function accepts
        accepted_args = inspect.signature(func).parameters.keys()
        filtered_args = {
            key: value for key, value in args.items() if key in accepted_args
        }

        return func(**filtered_args)

    # @retry(wait=wait_random_exponential(multiplier=1, max=40), stop=stop_after_attempt(3))
    def chat_completion_request(self, function_call="auto", model=GPT_MODEL):
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=self.memory_store.messages,
                functions=self.functions,
                function_call=function_call,
            )
            return response
        except Exception as e:
            logger.error("Unable to generate ChatCompletion response")
            logger.error(f"Exception: {e}")
            raise

    def append_function_result_message(self, function_call_name, results):
        self._append_message(
            {"role": "function", "content": results, "name": function_call_name}
        )

    def process_messages(self):
        # TODO: make ending function name settable OR move this into the childclass
        iter_count = 0
        function_call_name = ""

        results = ""

        while (
            iter_count < self.threshold
            and function_call_name != self.terminating_function_call_name
        ):
            chat_response = self.chat_completion_request()
            assistant_message = chat_response["choices"][0]["message"]
            self._append_message(assistant_message.to_dict_recursive())
            logger.debug(assistant_message)
            if "function_call" in assistant_message:
                results = self.execute_function_call(assistant_message)
                function_call_name = assistant_message["function_call"]["name"]
                self.append_function_result_message(function_call_name, results)
            else:
                self._append_message({"role": "user", "content": "Continue"})
            iter_count += 1

        if function_call_name == self.terminating_function_call_name:
            return results
        raise Exception(
            "I had to stop the search loop before plan for formulated because I reached the end of my allotted function calls"
        )
