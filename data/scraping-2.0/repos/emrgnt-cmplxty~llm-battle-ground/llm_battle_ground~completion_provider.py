"""Module for the completion provider"""
import textwrap
from enum import Enum
from typing import Optional

from automata.llm import OpenAIChatCompletionProvider, OpenAIConversation

from llm_battle_ground.models import make_model
from llm_battle_ground.types import LLMProviders


class RunMode(Enum):
    """Specifies the mode of running the completion provider"""

    SIMILARITY = "similarity"
    VANILLA_ZERO_SHOT = "vanilla-zero-shot"


class CompletionProvider:
    """Concrete class for completion providers"""

    def __init__(
        self,
        run_mode: RunMode,
        model: str,
        temperature: float,
        provider: LLMProviders,
    ):
        self.run_mode = run_mode
        self.provider = provider
        self.model = model
        self.temperature = temperature
        if self.provider == LLMProviders.OPENAI:
            self.completion_instance = OpenAIChatCompletionProvider(
                model=self.model,
                temperature=self.temperature,
                stream=True,
                conversation=OpenAIConversation(),
                functions=[],
            )

        elif self.provider:
            # means we need to load the model locally
            # TODO: batch size
            self.completion_instance = make_model(
                provider=self.provider.value,
                name=self.model,
                batch_size=1,
                temperature=temperature,
            )

    def get_completion(self, **kwargs) -> str:
        """Returns the raw and cleaned completions for the given prompt"""
        code_snippet = kwargs.get("code_snippet")
        if not isinstance(code_snippet, str):
            raise ValueError("Code snippet must be provided as a string.")

        if self.run_mode not in [
            RunMode.SIMILARITY,
            RunMode.VANILLA_ZERO_SHOT,
        ]:
            raise ValueError("No such run mode.")
        vanilla_instructions = self.get_formatted_instruction(**kwargs)
        return self.generate_vanilla_completion(
            vanilla_instructions,
            code_snippet=code_snippet,
        )

    def generate_vanilla_completion(
        self, instructions: str, code_snippet: Optional[str] = None
    ) -> str:
        """Generates a vanilla completion for the given prompt"""
        if self.provider == LLMProviders.OPENAI:
            assert isinstance(
                self.completion_instance, OpenAIChatCompletionProvider
            )
            return self.completion_instance.standalone_call(instructions)
        elif self.provider == LLMProviders.HUGGING_FACE:
            # TODO - Add assertion to protect against faulty instance
            # e.g. assert isinstnace(...)
            return f"{code_snippet}{self.completion_instance.codegen(instructions, num_samples=1)[0]}"
        else:
            raise ValueError("No such provider.")

    def get_perplexity(self, prefix: str, completion: str) -> float:
        """Returns the perplexity of the completion for the given prompt"""
        if self.provider == LLMProviders.HUGGING_FACE:
            if self.model in [
                "wizardcoder",
                "platypus",
                "mpt-instruct",
                "falcon-instruct",
                "stablebeluga",
            ]:  # some models are instruction based.
                return self.completion_instance.perplexity(prefix, completion)
            else:
                # remove first Example
                return self.completion_instance.perplexity(prefix, completion[7:])
        else:
            raise ValueError(
                "No such provider or provider does not support perplexity."
            )

    def get_formatted_instruction(
        self,
        **kwargs,
    ) -> str:
        """Formats the instruction for the given prompt"""

        if self.run_mode == RunMode.SIMILARITY:
            task_input = kwargs.get("task_input")
            num_forward_examples = kwargs.get("num_forward_examples")
            if not task_input or not num_forward_examples:
                raise ValueError("Missing required arguments.")
            if self.provider in [LLMProviders.HUGGING_FACE]:
                if self.model in [
                    "wizardcoder",
                    "platypus",
                    "mpt-instruct",
                    "falcon-instruct",
                    "stablebeluga",
                ]:  # some models are instruction based.
                    return textwrap.dedent(
                        """
                        Closely examine the following examples -

                        Input:
                        {TASK_INPUT}

                        Now, use those examples to predict the next {NUM_FORWARD_EXAMPLES} examples that will follow. DO NOT OUTPUT ANY ADDITIONAL TEXT, ONLY THE NEXT {NUM_FORWARD_EXAMPLES} EXAMPLES.
                        """
                    ).format(
                        TASK_INPUT=task_input,
                        NUM_FORWARD_EXAMPLES=num_forward_examples,
                    )
                else:
                    return textwrap.dedent(
                        """
                        {TASK_INPUT}
                        Example"""
                    ).format(TASK_INPUT=task_input)
            else:
                return textwrap.dedent(
                    """
                    Closely examine the following examples -
    
                    Input:
                    {TASK_INPUT}
    
                    Now, use those examples to predict the next {NUM_FORWARD_EXAMPLES} examples that will follow. DO NOT OUTPUT ANY ADDITIONAL TEXT, ONLY THE NEXT {NUM_FORWARD_EXAMPLES} EXAMPLES.
                    Output:
                    """
                ).format(
                    TASK_INPUT=task_input,
                    NUM_FORWARD_EXAMPLES=num_forward_examples,
                )
        elif self.run_mode == RunMode.VANILLA_ZERO_SHOT:
            task_input = kwargs.get("task_input")
            code_snippet = kwargs.get("code_snippet")
            if not task_input or not code_snippet:
                raise ValueError("Missing required arguments.")

            # if hugging-face we use a simplified version of the instructions
            if self.provider in [LLMProviders.HUGGING_FACE]:
                return textwrap.dedent(
                    """
        ### Introduction:
        {TASK_INPUT}
        
        {STARTING_CODE}\n"""
                ).format(
                    TASK_INPUT=task_input,
                    CODE_PROMPT=code_snippet,
                    STARTING_CODE=code_snippet,
                )
            else:
                return textwrap.dedent(
                    """
        ### Introduction:
        {TASK_INPUT}
    
        ### Instruction:
        Provide a response which completes the following Python code: 
    
        code:
        ```python
        {CODE_PROMPT}
        ```
    
        ### Notes: 
        Respond with the entire complete function definition, including a re-stated function definition.
        Use only built-in libraries and numpy, assume no additional imports other than those provided and 'from typings import *'.
        Optimize your algorithm to run as efficiently as possible. This is a Hard LeetCode problem, and so in the vast majority of cases
        the appropriate solution will run in NlogN or faster. Lastly, start by re-stating the given tests into
        the local python environment, and ensure that your final solution passes all given tests. 
    
        ### Result:
        When you have completed the problem or have ran out of alotted iterations or tokens, return a markdown-snippet with your final algorithmic implementation using `call_termination`. 
        E.g. ```python\n{CODE_PROMPT}\n  #.... (Code Continued) ...```
        Your final result should follow EXACTLY the format shown above, except for additional imports which may be added.
    
                        """
                ).format(TASK_INPUT=task_input, CODE_PROMPT=code_snippet)

        else:
            raise ValueError("No such run mode.")
