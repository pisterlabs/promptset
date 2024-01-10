import pytest  # noqa: F401
import asyncio
from typing import List, Any, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from prompt_breeder.types import UnitOfEvolution, Population
from prompt_breeder.prompts.string import (
    StringPrompt,
    StringTaskPrompt,
    StringMutationPrompt,
    StringThinkingStyle,
)
from prompt_breeder.mutators.zero_order_hypermutation import (
    ZeroOrderHypermutation,
)


class MockPassthroughLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "passthrough"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return prompt


class MockThinkingStyleProvider:
    def __iter__(self):
        return self

    def __next__(self):
        return StringThinkingStyle(text="Lets think about this.")


def test_runs_over_unit():
    llm = MockPassthroughLLM()
    prompt0 = StringTaskPrompt(text="Solve the math word problem, show your workings.")
    prompt1 = StringTaskPrompt(text="Solve the math word problem.")
    unit = UnitOfEvolution(
        problem_description=StringPrompt(
            text="Solve the math word problem, giving your answer as an arabic numeral"
        ),
        task_prompt_set=[
            prompt0,
            prompt1,
        ],
        mutation_prompt=StringMutationPrompt(text="make the task better."),
        elites=[],
    )
    mutator = ZeroOrderHypermutation.from_llm(
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        thinking_style_provider=MockThinkingStyleProvider(),
        llm=llm,
        verbose=1,
    )
    population = Population(members=[unit])
    _ = mutator.mutate(population, unit)
    _ = asyncio.run(mutator.amutate(population, unit))
