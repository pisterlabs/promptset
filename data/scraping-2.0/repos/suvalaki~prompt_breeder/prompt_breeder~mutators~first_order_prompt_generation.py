from typing import Callable
from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLanguageModel
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema.messages import SystemMessage
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model

from prompt_breeder.types import MutationPrompt, TaskPrompt
from prompt_breeder.mutators.base import DirectMutator


BASE_TEMPLATE = PromptTemplate.from_template(
    "{mutation_prompt}  INSTRUCTION: {task_prompt}  INSTRUCTION MUTATNT: "
)
CHAT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a meta heuristic assisting in the development of "
            "better instructions to complete a task. Generate a new improved "
            "instruction mutant to complete the task."
        ),
        HumanMessagePromptTemplate.from_template(
            "{mutation_prompt}  INSTRUCTION: {task_prompt}  INSTRUCTION MUTATNT: "
        ),
    ]
)
PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=BASE_TEMPLATE,
    conditionals=[(is_chat_model, CHAT_TEMPLATE)],
)


class FirstOrderPromptGeneration(LLMChain, DirectMutator):
    """We concatenate the mutation-prompt (red), to the parent
    task-prompt (blue), and pass it to the LLM to produce the mutated task-prompt.

    This procedure is identical to the initialization method, except that a randomly
    sampled thinking-style string is not used. First-order prompt generation is
    Promptbreeder’s standard asexual mutation operat
    """

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
            mutation_prompt_factory=mutation_prompt_factory,
            task_prompt_factory=task_prompt_factory,
            prompt=PROMPT_SELECTOR.get_prompt(llm),
            **kwargs,
        )
