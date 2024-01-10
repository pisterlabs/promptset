from typing import Callable, Iterator
from langchain.llms.base import BaseLanguageModel
from langchain.chains.llm import LLMChain
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema.messages import SystemMessage
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model

from prompt_breeder.types import TaskPrompt, MutationPrompt, ThinkingStyle
from prompt_breeder.mutators.base import Hypermutation
from prompt_breeder.mutators.first_order_prompt_generation import (
    FirstOrderPromptGeneration,
)

BASE_TEMPLATE = PromptTemplate.from_template(
    "Please summarize and improve the following instruction: {mutation_prompt} "
)
CHAT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a meta heuristic assisting in the development of "
            "better instructions to complete a task. Generate a new improved "
            "insutrction mutant to complete the task."
        ),
        HumanMessagePromptTemplate.from_template(
            "Please summarize and improve the following instruction: {mutation_prompt} "
        ),
    ]
)
PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=BASE_TEMPLATE,
    conditionals=[(is_chat_model, CHAT_TEMPLATE)],
)


class FirstOrderMutation(LLMChain):
    @classmethod
    def from_llm(cls, llm: BaseLanguageModel, **kwargs):
        return cls(
            llm=llm,
            prompt=PROMPT_SELECTOR.get_prompt(llm),
            **kwargs,
        )


class FirstOrderHypermutation(Hypermutation):
    """Concatenate the hyper-mutation-prompt "Please summarize
    and improve the following instruction:" to a mutation-prompt so that the LLM gener-
    ates a new mutation-prompt. This newly generated mutation-prompt is then applied to
    the taskprompt of that unit (see First-Order Prompt Generation)."""

    mutate_mutator_chain: FirstOrderMutation
    mutate_task_prompt_chain: FirstOrderPromptGeneration

    @classmethod
    def from_llm(
        cls,
        mutation_prompt_factory: Callable[[str], MutationPrompt],
        task_prompt_factory: Callable[[str], TaskPrompt],
        thinking_style_provider: Iterator[ThinkingStyle],
        llm: BaseLanguageModel,
        **kwargs
    ):
        return cls(
            task_prompt_factory=task_prompt_factory,
            mutation_prompt_factory=mutation_prompt_factory,
            thinking_style_provider=thinking_style_provider,
            mutate_mutator_chain=FirstOrderMutation.from_llm(
                llm=llm,
                **kwargs,
            ),
            mutate_task_prompt_chain=FirstOrderPromptGeneration.from_llm(
                llm=llm,
                task_prompt_factory=task_prompt_factory,
                mutation_prompt_factory=mutation_prompt_factory,
                **kwargs,
            ),
            **kwargs,
        )
