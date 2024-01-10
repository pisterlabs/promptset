import pytest  # noqa: F401
import asyncio
from typing import Dict, List, Any, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from prompt_breeder.types import UnitOfEvolution, Population
from prompt_breeder.prompts.string import (
    StringPrompt,
    StringTaskPrompt,
    StringMutationPrompt,
)
from prompt_breeder.mutators.first_order_prompt_generation import (
    FirstOrderPromptGeneration,
)


KEY0 = "key0"
KEY1 = "key1"

# "{task_prompt_set}  INSTRUCTION MUTATNT: "
FIXED_PROMPT_REPLY: Dict[str, str] = {
    KEY0: KEY0,
    KEY1: KEY1,
}


class MockLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom_first_order_gen"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        for k in FIXED_PROMPT_REPLY:
            if k in prompt:
                return FIXED_PROMPT_REPLY[k]
        raise ValueError(f"key isnt available. prompt: {prompt}")


def test_runs_over_unit():
    llm = MockLLM()
    prompt0 = StringTaskPrompt(text=KEY0)
    prompt1 = StringTaskPrompt(text=KEY1)
    unit = UnitOfEvolution(
        problem_description=StringPrompt(text="ignored by first order"),
        task_prompt_set=[
            prompt0,
            prompt1,
        ],
        mutation_prompt=StringMutationPrompt(text="not ignored but also not needed"),
        elites=[],
    )
    mutator = FirstOrderPromptGeneration.from_llm(
        llm=llm,
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        verbose=1,
    )
    population = Population(members=[unit])
    _ = mutator.mutate(population, unit)

    _ = asyncio.run(mutator.amutate(population, unit))
