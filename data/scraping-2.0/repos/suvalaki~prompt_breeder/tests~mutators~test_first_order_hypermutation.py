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
    StringThinkingStyle,
)
from prompt_breeder.mutators.first_order_hypermutation import (
    FirstOrderHypermutation,
)


KEY0 = "key0"
KEY1 = "key1"
KEY2 = "key2"
KEY3 = "key3"

# "{task_prompt_set}  INSTRUCTION MUTATNT: "
FIXED_PROMPT_REPLY: Dict[str, str] = {
    KEY0: KEY0,
    KEY1: KEY1,
    KEY2: KEY2,
    KEY3: KEY3,
}


class MockLLM(LLM):
    @property
    def _llm_type(self) -> str:
        return "custom_first_order_hyper"

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


class MockThinkingStyleProvider:
    def __iter__(self):
        return self

    def __next__(self):
        return StringThinkingStyle(text="Lets think about this.")


# Starts by calling the LLM on the mutation prompt (key3)
# then calls first order prompt gen on key0, key1 and key3 (key0, key1 first)
def test_runs_over_unit():
    llm = MockLLM()
    prompt0 = StringTaskPrompt(text=KEY0)
    prompt1 = StringTaskPrompt(text=KEY1)
    unit = UnitOfEvolution(
        problem_description=StringPrompt(text=KEY2),
        task_prompt_set=[
            prompt0,
            prompt1,
        ],
        mutation_prompt=StringMutationPrompt(text=KEY3),
        elites=[],
    )
    mutator = FirstOrderHypermutation.from_llm(
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        thinking_style_provider=MockThinkingStyleProvider(),
        llm=llm,
        verbose=1,
    )
    population = Population(members=[unit])
    ans = mutator.mutate(population, unit)
    assert all([str(x) in [KEY0, KEY1] for x in ans.task_prompt_set])

    ans = asyncio.run(mutator.amutate(population, unit))
    assert all([str(x) in [KEY0, KEY1] for x in ans.task_prompt_set])
