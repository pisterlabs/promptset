import os

import openai as openai

from .code_manager.abstract_extractor import LanguageHandler
from .openai_service import GPT_3_MODELS, GPT_4_MODELS, num_tokens_from_messages


class TestGenerator:
    TEMPERATURE = 0.4  # temperature = 0 can sometimes get stuck in repetitive loops, so we use 0.4

    def __init__(
        self,
        function_to_test: str,
        language: str,
        unit_test_package: str,
        debug: bool = False,
        approx_min_cases_to_cover: int = 7,
        reruns_if_fail: int = 1,
        use_gpt_4: bool = False,
        openai_api_key: str = None,
    ):
        openai.api_key = (
            openai_api_key if openai_api_key else os.environ["OPENAI_API_KEY"]
        )
        self.messages = []
        self.language = language
        self.unit_test_package = unit_test_package
        self.function_to_test = function_to_test
        self.debug = debug
        self.approx_min_cases_to_cover = approx_min_cases_to_cover
        self.reruns_if_fail = reruns_if_fail
        self.code_handler = LanguageHandler[language.upper()].value()
        self.model_set = GPT_4_MODELS if use_gpt_4 else GPT_3_MODELS

    def create_gpt_message(self, role: str, content: str) -> dict:
        message = {"role": role, "content": content}
        if role == "system":
            messages_without_sys_message = [
                m for m in self.messages if m["role"] != "system"
            ]
            self.messages = [message] + messages_without_sys_message
        else:
            self.messages.append(message)

    color_prefix_by_role = {
        "system": "\033[0m",  # gray
        "user": "\033[0m",  # gray
        "assistant": "\033[92m",  # green
    }

    def print_messages(self, messages) -> None:
        """Prints messages sent to or from GPT."""
        for message in messages:
            role = message["role"]
            color_prefix = self.color_prefix_by_role[role]
            content = message["content"]
            print(f"{color_prefix}\n[{role}]\n{content}")

    def print_message_delta(self, delta) -> None:
        """Prints a chunk of messages streamed back from GPT."""
        if "role" in delta:
            role = delta["role"]
            color_prefix = self.color_prefix_by_role[role]
            print(f"{color_prefix}\n[{role}]\n", end="")
        elif "content" in delta:
            content = delta["content"]
            print(content, end="")
        else:
            pass

    def get_explanation_of_function(self) -> str:
        self.create_gpt_message(
            "system",
            f"You are a world-class {self.language} developer with an eagle eye for unintended bugs and edge cases. ...",
        )
        self.create_gpt_message(
            "user",
            f"""Please explain the following {self.language} function. Review what each element of the function is doing precisely and what the author's intentions may have been. Organize your explanation as a markdown-formatted, bulleted list.

```{self.language}
{self.function_to_test}
```""",
        )
        return self.generate_stream_response()

    def get_assistant_stream_response(self, api_response: dict) -> str:
        assistant_message = ""
        for chunk in api_response:
            delta = chunk["choices"][0]["delta"]
            if self.debug:
                self.print_message_delta(delta)
            if "content" in delta:
                assistant_message += delta["content"]
        return assistant_message

    def find_gpt3_model(self):
        num_tokens = num_tokens_from_messages(self.messages)
        for max_tokens, model in self.model_set.items():
            if num_tokens < max_tokens:
                return model
        raise Exception(f"Too many tokens ({num_tokens}) for {model}")

    def generate_stream_response(self) -> str:
        model = self.find_gpt3_model()
        response = openai.ChatCompletion.create(
            model=model,
            messages=self.messages,
            temperature=self.TEMPERATURE,
            stream=True,
        )
        assistant_message = self.get_assistant_stream_response(response)
        self.create_gpt_message("assistant", assistant_message)
        return assistant_message

    def generate_plan(self) -> str:
        self.create_gpt_message(
            "user",
            f"""A good unit test suite should aim to:
- Test the function's behavior for a wide range of possible inputs
- Test edge cases that the author may not have foreseen
- Take advantage of the features of `{self.unit_test_package}` to make the tests easy to write and maintain
- Be easy to read and understand, with clean code and descriptive names
- Be deterministic, so that the tests always pass or fail in the same way

To help unit test the function above, list diverse scenarios that the function should be able to handle (and under each scenario, include a few examples as sub-bullets).""",
        )
        plan = self.generate_stream_response()

        # Set if further elaboration is needed
        num_bullets = max(plan.count("\n-"), plan.count("\n*"))
        self.elaboration_needed = num_bullets < self.approx_min_cases_to_cover

        return plan

    def generate_elaboration(self) -> str:
        self.create_gpt_message(
            "user",
            f"""In addition to those scenarios above, list a few rare or unexpected edge cases (and as before, under each edge case, include a few examples as sub-bullets).""",
        )
        return self.generate_stream_response()

    def generate_unit_test(self) -> str:
        package_comment = ""
        if self.unit_test_package == "pytest":
            package_comment = "# below, each test case is represented by a tuple passed to the @pytest.mark.parametrize decorator"

        self.create_gpt_message(
            "system",
            "You are a world-class Python developer with an eagle eye for unintended bugs and edge cases. You write careful, accurate unit tests. When asked to reply only with code, you write all of your code in a single block.",
        )
        self.create_gpt_message(
            "user",
            f"""Using {self.language} and the `{self.unit_test_package}` package, write a suite of unit tests for the function, following the cases above. Include helpful comments to explain each line. Reply only with code, formatted as follows:

```{self.language}
# imports
import {self.unit_test_package}  # used for our unit tests
{{insert other imports as needed}}

# function to test
{self.function_to_test}

# unit tests
{package_comment}
{{insert unit test code here}}
```""",
        )
        return self.generate_stream_response()

    def unit_tests_from_function(
        self,
    ) -> str:
        self.get_explanation_of_function()
        self.generate_plan()
        if self.elaboration_needed:
            self.generate_elaboration()
        generated_tests = self.generate_unit_test()

        # handle errors
        # check the output for errors
        cleaned_tests = (
            generated_tests.split(f"```{self.language}")[1].split("```")[0].strip()
        )
        try:
            self.code_handler.is_valid_code(
                cleaned_tests
            )  # TODO: use tree-sitter for valdation instead
        except SyntaxError as e:
            if self.reruns_if_fail > 0:
                return self.unit_tests_from_function(
                    function_to_test=self.function_to_test,
                    reruns_if_fail=self.reruns_if_fail - 1,
                )
            raise
        return cleaned_tests
