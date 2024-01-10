# Create a simple one mutator binary tourno
import pytest  # noqa: F401
import asyncio
from typing import List, Any, Optional
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from prompt_breeder.evolution.binary_tournament import BinaryEvolution


from prompt_breeder.types import UnitOfEvolution, Population
from prompt_breeder.prompts.string import (
    StringPrompt,
    StringTaskPrompt,
    StringMutationPrompt,
    StringThinkingStyle,
)
from prompt_breeder.evolution.fitness import BestMemberFitness

from prompt_breeder.mutators.zero_order_hypermutation import (
    ZeroOrderHypermutation,
)
from prompt_breeder.evolution.base import EvolutionExecutor
from prompt_breeder.mutators.elite import (
    AddElite,
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


class StringLengthFitness:
    def score(self, prompt: StringTaskPrompt, **kwargs) -> int:
        return len(str(prompt))

    def ascore(self, prompt: StringTaskPrompt, **kwargs) -> int:
        return len(str(prompt))


class MockThinkingStyleProvider:
    def __iter__(self):
        return self

    def __next__(self):
        return StringThinkingStyle(text="Lets think about this.")


def test_one_mutant_unit():
    # Create an initial popoulation. Lets just use 2 units
    # By string length unit0 should always have the higher fitness
    prompt00 = StringTaskPrompt(
        text="one Solve the math word problem, show your workings.     "
    )
    prompt01 = StringTaskPrompt(text="one Solve the math word problem.      ")
    unit0 = UnitOfEvolution(
        problem_description=StringPrompt(
            text="Solve the math word problem, giving your answer as an arabic numeral."
        ),
        task_prompt_set=[
            prompt00,
            prompt01,
        ],
        mutation_prompt=StringMutationPrompt(text="make the task better."),
        elites=[],
    )

    prompt10 = StringTaskPrompt(
        text="two Solve the math word problem, show your workings."
    )
    prompt11 = StringTaskPrompt(text="two Solve the math word problem.")
    unit1 = UnitOfEvolution(
        problem_description=StringPrompt(
            text="Solve the math word problem, giving your answer as an arabic numeral."
        ),
        task_prompt_set=[
            prompt10,
            prompt11,
        ],
        mutation_prompt=StringMutationPrompt(text="make the task better."),
        elites=[],
    )

    population = Population(members=[unit0, unit1])

    # Create the set of mutators
    llm = MockPassthroughLLM()
    mutator = ZeroOrderHypermutation.from_llm(
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        thinking_style_provider=MockThinkingStyleProvider(),
        llm=llm,
        verbose=1,
    )

    multiple_scorer = BestMemberFitness(scorer=StringLengthFitness())

    evolution_step = BinaryEvolution(
        fitness_scorer=multiple_scorer,
        pre_step_modifiers=[],
        mutators=[mutator],
        post_step_modifiers=[],
        verbose=1,
    )

    # Run the step
    ans = evolution_step.run({"population": population})

    # One of the UnitsOfEvoluion will have its prompts replaced with mutated
    # versions of the others.
    assert sum([x == unit0 or x == unit1 for x in ans.members]) == 1


def test_multiple_steps():
    # Create an initial popoulation. Lets just use 2 units
    # By string length unit0 should always have the higher fitness
    prompt00 = StringTaskPrompt(
        text="one Solve the math word problem, show your workings.     "
    )
    prompt01 = StringTaskPrompt(text="one Solve the math word problem.      ")
    unit0 = UnitOfEvolution(
        problem_description=StringPrompt(
            text="Solve the math word problem, giving your answer as an arabic numeral."
        ),
        task_prompt_set=[
            prompt00,
            prompt01,
        ],
        mutation_prompt=StringMutationPrompt(text="make the task better."),
        elites=[],
    )

    prompt10 = StringTaskPrompt(
        text="two Solve the math word problem, show your workings."
    )
    prompt11 = StringTaskPrompt(text="two Solve the math word problem.")
    unit1 = UnitOfEvolution(
        problem_description=StringPrompt(
            text="Solve the math word problem, giving your answer as an arabic numeral."
        ),
        task_prompt_set=[
            prompt10,
            prompt11,
        ],
        mutation_prompt=StringMutationPrompt(text="make the task better."),
        elites=[],
    )

    population = Population(members=[unit0, unit1])

    # Create the set of mutators
    llm = MockPassthroughLLM()
    mutator = ZeroOrderHypermutation.from_llm(
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        thinking_style_provider=MockThinkingStyleProvider(),
        llm=llm,
        verbose=1,
    )
    elite_mutator = AddElite(
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        fitness_scorer=StringLengthFitness(),
        verbose=1,
    )

    multiple_scorer = BestMemberFitness(scorer=StringLengthFitness())

    evolution_step = BinaryEvolution(
        fitness_scorer=multiple_scorer,
        pre_step_modifiers=[],
        mutators=[mutator],
        post_step_modifiers=[elite_mutator],
        verbose=1,
    )

    evolution = EvolutionExecutor(step=evolution_step)
    ans = evolution.run({"population": population, "generations": 2})
    assert all([len(x.elites) == 2 for x in ans.members])

    evolution = EvolutionExecutor(step=evolution_step, return_intermediate_steps=True)
    ans = evolution.run({"population": population, "generations": 2})
    assert len(ans) == 2

    evolution = EvolutionExecutor(step=evolution_step, return_intermediate_steps=True)
    ans = asyncio.run(evolution.arun({"population": population, "generations": 2}))
    assert len(ans) == 2
