from abc import ABC, abstractmethod
from functools import lru_cache
from itertools import chain
import ast
import doctest
import functools
import os
import random
import sys

import openai
import tiktoken
import pytest

from .data import FakeFunctionDefinition


# TODO: think. Is it useful to pass the failure to this callback?
def RAISE_EXCEPTION():
    raise ValueError("GPT returned bad output")


def RANDOM_SELECTOR(examples, call_args, call_kwargs, min_examples=None):
    return random.sample(
        examples,
        k=len(examples) if min_examples is None else min(len(examples), min_examples),
    )


# TODO: obtain and set with a context manager
openai.api_key = os.getenv("OPENAI_API_KEY")


class Runner(ABC):
    def __init__(
        self,
        function,
        override_name=None,
        on_api_error=RAISE_EXCEPTION,
        on_invalid_response=RAISE_EXCEPTION,
        external_example_file=None,
        num_examples=None,
        example_selector=RANDOM_SELECTOR,
        **api_kwargs,
    ):
        self.name = function.__name__ if override_name is None else override_name
        self.on_api_error = on_api_error
        self.on_invalid_response = on_invalid_response

        # Examples can be provided from an external `external_example_file` or as a docstring body.
        examples = None
        if external_example_file is not None:
            with open(external_example_file) as example_file:
                examples = example_file.read()
        self.definition = FakeFunctionDefinition.from_docstring(
            function.__doc__, external_examples=examples
        )
        self.num_examples = num_examples
        self.example_selector = example_selector

        self.api_kwargs = dict()
        try:
            for k, v in api_kwargs.items():
                assert k.startswith(
                    "api_"
                ), f"Invalid parameter {k!r}. Extra API kwargs must be prefixed with 'api_'"
                self.api_kwargs[k[4:]] = v
        except AssertionError as exc:
            raise ValueError from exc

    @abstractmethod
    def calculate_prompt_tokens(self, prompt):
        """
        Return the number of tokens for this particular prompt.

        """
        pass

    @abstractmethod
    def calculate_text_tokens(self, text):
        """
        Return the number of tokens for this particular arbitrary text.

        """
        pass

    @abstractmethod
    def make_prompt(self, *args, _examples=None, **kwargs):
        """
        Return the prompt to use for the API for the given set of examples.

        This prompt will be used to generate the output of the function.

        """
        pass

    @abstractmethod
    def call_api(self, prompt):
        """
        Call the API with the given prompt returning the raw response.

        """
        pass

    @abstractmethod
    def api_response_to_text(self, response):
        """
        Return the part of the response representing the model text completion.

        """
        pass

    def calculate_tokens_per_call(self, *args, **kwargs):
        """
        Return the number of tokens per call.

        Depending on the runner, this can be an exact value or an average value.

        Example: {"result_type": "exact", "value": 100}

        """
        prompt_tokens = None
        if self.num_examples is None:  # In this case we can provide an exact answer
            prompt_tokens = {
                "result_type": "exact",
                "value": self.calculate_prompt_tokens(
                    self.make_prompt(*args, **kwargs)
                ),
            }
        else:  # We only can approximate the number of tokens per call by sampling
            prompt_tokens = {
                "result_type": "average",
                "value": sum(
                    self.calculate_prompt_tokens(self.make_prompt(*args, **kwargs))
                    for _ in range(1000)
                )
                / 1000,
            }
        response_tokens = {
            "result_type": "average",
            "value": sum(
                self.calculate_text_tokens(example.want)
                for example in self.definition.examples
            )
            / len(self.definition.examples),
        }

        return {"prompt": prompt_tokens, "response": response_tokens}

    def _deserialize_completion(self, completion):
        try:
            return ast.literal_eval(completion)
        except Exception as gpt_exception:
            try:
                return self.on_invalid_response()
            except Exception as user_exception:
                raise user_exception from gpt_exception

    def __call__(self, *args, **kwargs):
        """Run this imaginary function with the mighty power of GPT."""

        prompt = self.make_prompt(*args, **kwargs)
        try:
            response = self.call_api(prompt)
        except Exception as api_exception:
            try:
                return self.on_api_error()
            except Exception as user_exception:
                raise user_exception from api_exception

        completion = self.api_response_to_text(response)

        return self._deserialize_completion(completion)

    def test_prompt_examples(self, *args, **kwargs):
        """
        This function returns a pytest parametrizer decorator that let you test
        one by one the examples provided in the docstring.

        This will not perform any call to OpenAI.

        """
        examples = list()
        for example in self.definition.examples:
            function_name = example.source.split("(")[0].strip()
            examples.append(
                (
                    function_name,
                    example.call_args_obj,
                    example.call_kwargs_obj,
                    example.want_obj,
                )
            )

        return pytest.mark.parametrize(
            "function_name,call_args,call_kwargs,return_value", examples
        )(*args, **kwargs)

    def test_task_generalization(self):
        """
        This function is a pytest test that let you test the ability to
        generalize the task with the examples given in the docstring.

        This works by prompting the model as many times as examples are in the
        definition, plucking out one example at a time and testing if that call
        returns the expected output for that example.

        Please note that THIS WILL PERFORM MANY CALLS to OpenAI's API.

        """

        def _make_test_prompts():
            for i in range(len(self.definition.examples)):
                preamble = (
                    self.definition.examples[:i] + self.definition.examples[i + 1 :]
                )
                missing = self.definition.examples[i]
                if missing.options.get(doctest.SKIP, None):
                    continue
                yield (
                    self.make_prompt(
                        *missing.call_args, _examples=preamble, **missing.call_kwargs
                    ),
                    missing.want_obj,
                )

        for i, (prompt, wanted) in enumerate(_make_test_prompts()):
            response = self.call_api(prompt)
            completion = self.api_response_to_text(response)
            current = self._deserialize_completion(completion)
            assert (
                current == wanted
            ), f"In test #{i}: {prompt}, and got {current!r} instead of {wanted!r}"


class CompletionAPIRunner(Runner):
    """Infere call result using OpenAI's completion API."""

    @property
    @lru_cache(maxsize=1)
    def tokenizer(self):
        return tiktoken.encoding_for_model(self.engine)

    @property
    def engine(self):
        return self.api_kwargs.get("engine", "text-davinci-003")

    def calculate_text_tokens(self, text):
        return len(self.tokenizer.encode(text))

    def calculate_prompt_tokens(self, prompt):
        return self.calculate_text_tokens(prompt)

    def make_prompt(self, *args, _examples=None, **kwargs):
        """Build the prompt for the given set of parameters."""

        doc = f">>> {self.name}.__doc__\n{self.definition.summary!r}"

        example_base = self.example_selector(
            _examples if _examples is not None else self.definition.examples,
            args, kwargs, min_examples=self.num_examples
        )
        if self.num_examples is not None:
            example_base = example_base[:min(self.num_examples, len(example_base))]
        examples = "\n".join(f">>> {e.source}\n{e.want}" for e in example_base)

        args = [repr(a) for a in args]
        kwargs = [f"{k}={v!r}" for k, v in kwargs.items()]
        call = f'>>> {self.name}({", ".join(args + kwargs)})'

        return "\n".join([doc, examples, call])

    def call_api(self, prompt):
        return openai.Completion.create(
            prompt=prompt, **{**{"engine": self.engine}, **self.api_kwargs}
        )

    def api_response_to_text(self, response):
        return response["choices"][0]["text"].strip()


class ChatCompletionAPIRunner(Runner):
    """
    Infere call result using OpenAI's chat API.

    In this runner a prompt is a structured message that follows the API
    described here:

    https://platform.openai.com/docs/api-reference/chat

    """

    @property
    def model(self):
        return self.api_kwargs.get("model", "gpt-3.5-turbo")

    @property
    @lru_cache(maxsize=1)
    def tokenizer(self):
        return tiktoken.encoding_for_model(self.model)

    def calculate_text_tokens(self, text):
        return len(self.tokenizer.encode(text))

    def calculate_prompt_tokens(self, prompt):
        """
        source: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

        """
        num_tokens = 0
        for message in prompt:
            num_tokens += (
                4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            )
            for key, value in message.items():
                num_tokens += self.calculate_text_tokens(value)
                if key == "name":  # if there's a name, the role is omitted
                    num_tokens += -1  # role is always required and always 1 token
            num_tokens += 2  # every reply is primed with <im_start>assistant
        return num_tokens

    def make_prompt(self, *args, _examples=None, **kwargs):
        """Build the prompt for the given set of parameters."""

        # <ominious-voice>You are a Python interpreter, you are a Python interpreter...</ominious-voice>
        python_prompt = [
            {
                "role": "system",
                "content": f'Python {sys.version} (main, Feb  7 2023, 12:19:31) [GCC 12.2.0] on {sys.platform}\nType "help", "copyright", "credits" or "license" for more information.',
            }
        ]

        # Show the function docstring summary
        doc = [
            {"role": "user", "content": f">>> {self.name}.__doc__"},
            {"role": "assistant", "content": f"{self.definition.summary!r}"},
        ]

        # Show some examples to ChatGPT
        example_base = self.example_selector(
            _examples if _examples is not None else self.definition.examples,
            args, kwargs, min_examples=self.num_examples
        )
        if self.num_examples is not None:
            example_base = example_base[:min(self.num_examples, len(example_base))]
        examples = [
            (
                {"role": "user", "content": f">>> {e.source}"},
                {"role": "assistant", "content": f"{e.want}"},
            )
            for e in example_base
        ]
        examples = list(chain.from_iterable(examples))

        # Ask about the current call
        args = [repr(a) for a in args]
        kwargs = [f"{k}={v!r}" for k, v in kwargs.items()]
        call = [
            {"role": "user", "content": f'>>> {self.name}({", ".join(args + kwargs)})'}
        ]

        return python_prompt + doc + examples + call

    def call_api(self, prompt):
        return openai.ChatCompletion.create(
            messages=prompt, **{**{"model": self.model}, **self.api_kwargs}
        )

    def api_response_to_text(self, response):
        return response["choices"][0]["message"]["content"].strip()


def _make_runner_decorator(runner):
    """Make a decorator that transform a function into a runner."""

    def runner_decorator(*args, **kwargs):
        """
        A decorator that transform a function without code but with a docstring
        containing examples into a function that calls some OpenAI API and
        perform few shot prompting on the examples.

        :param f: The function to transform.
        :param on_failure: A function to call if the model fails to return a valid Python output.
        :param engine: The OpenAI engine to use.
        :param external_example_file: A path to a file containing external examples instead of using the docstring.
        :param num_examples: The number of examples to use. If None, all examples are used.
        :param api_kwargs: Additional keyword arguments to pass to the OpenAI API.

        """
        if kwargs:

            def wrapper(f):
                return functools.wraps(f)(runner(f, **kwargs))

            return wrapper
        else:
            return functools.wraps(args[0])(runner(args[0]))

    return runner_decorator


gpt3run = _make_runner_decorator(CompletionAPIRunner)
chatgptrun = _make_runner_decorator(ChatCompletionAPIRunner)
