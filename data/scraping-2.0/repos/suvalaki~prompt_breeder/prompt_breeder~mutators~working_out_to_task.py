import asyncio
from typing import Iterator, Callable, List
from copy import deepcopy
from langchain.llms.base import BaseLanguageModel
from langchain.chains.llm import LLMChain
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema.messages import SystemMessage
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model

from prompt_breeder.types import (
    Phenotype,
    UnitOfEvolution,
    Population,
)
from prompt_breeder.mutators.base import Mutator
from prompt_breeder.types import MutationPrompt, TaskPrompt

# from langchain.prompts.example_selector.base import BaseExampleSelector

BASE_TEMPLATE = PromptTemplate.from_template(
    "I gave a friend an instruction and some advice. "
    "Here are the correct examples of his workings out: \n{context}\n"
    "The instruction was: "
)
CHAT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a meta heuristic assisting in the development of "
            "better instructions to complete a task. Generate a new improved "
            "insutrction mutant to complete the task."
        ),
        HumanMessagePromptTemplate.from_template(
            "I gave a friend an instruction and some advice. "
            "Here are the correct examples of his workings out: \n{context}\n"
            "The instruction was: "
        ),
    ]
)
PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=BASE_TEMPLATE,
    conditionals=[(is_chat_model, CHAT_TEMPLATE)],
)


# Lamarkin Operator
class WorkingOutToTask(LLMChain, Mutator):
    """Fill up a few-shot context with only workings out that led to correct answers.
    During evaluation we provide this few shot-context before the task-prompt, providing
    guidance as to the form of the working out that is desired. If the few-shot context
    list is full, a single randomly sampled new correct working out replaces an existing
    working out from the list after fitness evaluation of a unit on a new set of
    questions. In addition, with a 10% chance we resample the whole context list with
    probability inverse to the maximum context list length"""

    # Get new correct workings
    correct_working_out_provider: Iterator[Phenotype]
    max_context_size: int = 2

    @classmethod
    def from_llm(
        cls,
        mutation_prompt_factory: Callable[[str], MutationPrompt],
        task_prompt_factory: Callable[[str], TaskPrompt],
        llm: BaseLanguageModel,
        **kwargs
    ):
        return cls(
            llm=llm,
            prompt=PROMPT_SELECTOR.get_prompt(llm),
            mutation_prompt_factory=mutation_prompt_factory,
            task_prompt_factory=task_prompt_factory,
            **kwargs
        )

    def mutate(
        self, population: Population, unit: UnitOfEvolution, **kwargs
    ) -> UnitOfEvolution:
        unit = deepcopy(unit)
        examples = [
            next(self.correct_working_out_provider)
            for i in range(self.max_context_size)
        ]

        unit.task_prompt_set = [
            self.task_prompt_factory(
                self.run(
                    {
                        "context": "\n\n".join([str(example) for example in examples]),
                    },
                    **kwargs
                )
            )
            for member in unit.task_prompt_set
        ]
        return unit

    async def _asingleton_task_prompt(self, examples: List[Phenotype], **kwargs):
        return self.task_prompt_factory(
            await self.arun(
                {
                    "context": "\n\n".join([str(example) for example in examples]),
                },
                **kwargs
            )
        )

    async def amutate(
        self, population: Population, unit: UnitOfEvolution, **kwargs
    ) -> UnitOfEvolution:
        unit = deepcopy(unit)
        examples = [
            next(self.correct_working_out_provider)
            for i in range(self.max_context_size)
        ]
        unit.task_prompt_set = await asyncio.gather(
            *[self._asingleton_task_prompt(examples) for member in unit.task_prompt_set]
        )
        return unit
