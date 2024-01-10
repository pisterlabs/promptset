"""
This file contains the necessary wrappers to use the Anthropic API as an option
in OpenAI Evals.
"""

from typing import Any, Optional, Union
from evals.api import CompletionFn, CompletionResult
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT

from evals.prompt.base import (
    OpenAICreateChatPrompt,
)
from evals.record import record_sampling

from evalugator.evals_completers.utils_prompt_converters import CustomCompletionPrompt

# from evals.utils.api_utils import (
#    openai_chat_completion_create_retrying,
#    openai_completion_create_retrying,
# )

anthropic = Anthropic()


class AnthropicBaseCompletionResult(CompletionResult):
    def __init__(self, raw_data: Any, prompt: Any):
        self.raw_data = raw_data
        self.prompt = prompt

    def get_completions(self) -> list[str]:
        raise NotImplementedError


class AnthropicCompletionResult(AnthropicBaseCompletionResult):
    def get_completions(self) -> list[str]:
        completions = []
        if self.raw_data and hasattr(self.raw_data, "completion"):
            completion_text = self.raw_data.completion
            if completion_text[0] == " ":
                completion_text = completion_text[1:]
                # NOTE: "Assistant:" is the end of the AI_PROMPT that Anthropic
                # wants to be passed in, but that means that the first character
                # will often be a space.
                # This is a hack to remove that space. (don't want to add space
                # later because adding a trailing space often makes LLMs worse)
            completions.append(completion_text)
        return completions


class AnthropicCompletionFn(CompletionFn):
    def __init__(
        self,
        model: Optional[str] = None,
        #        api_base: Optional[str] = None,
        #        api_key: Optional[str] = None,
        #        n_ctx: Optional[int] = None,
        extra_options: dict = {},
        max_tokens: int = 10,
        manual_prompt: bool = False,
        **kwargs,
    ):
        self.model = model if model is not None else "claude-2"
        #        self.api_base = api_base
        #        self.api_key = api_key
        #        self.n_ctx = n_ctx
        self.extra_options = extra_options
        self.max_tokens = max_tokens
        self.manual_prompt = manual_prompt

    def __call__(
        self,
        prompt: Union[str, OpenAICreateChatPrompt],
        **kwargs,
    ) -> AnthropicCompletionResult:
        # this will get a string (look at chat_prompt_to_text_prompt in evals/prompt/base.py)
        prompt = CustomCompletionPrompt(prompt).to_formatted_prompt(
            custom_role_to_prefix={"system": "", "user": "", "assistant": ""},
            for_completion=False,  # suppress addition of "Assistant: " regardless of dictionary
        )
        if not self.manual_prompt:
            prompt = f"{HUMAN_PROMPT} {prompt}{AI_PROMPT}"[1:]  # remove initial \n

        assert isinstance(
            prompt, str
        ), f"Got type {type(prompt)}, with val {prompt} for prompt, expected str"

        # print(kwargs)
        # print(self.extra_options)
        result = anthropic.completions.create(
            model=self.model,
            prompt=prompt,
            max_tokens_to_sample=self.max_tokens,
            # **{**kwargs, **self.extra_options},
        )

        result = AnthropicCompletionResult(raw_data=result, prompt=prompt)
        record_sampling(prompt=result.prompt, sampled=result.get_completions())
        return result
