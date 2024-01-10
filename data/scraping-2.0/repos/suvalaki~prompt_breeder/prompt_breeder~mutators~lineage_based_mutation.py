from typing import List, Dict, Any, Optional, Tuple, Callable

from langchain.llms.base import BaseLanguageModel
from langchain.chains.llm import LLMChain, PromptValue
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema.messages import SystemMessage
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.callbacks.manager import (
    CallbackManagerForChainRun,
    AsyncCallbackManagerForChainRun,
)

from prompt_breeder.types import MutationPrompt, TaskPrompt
from prompt_breeder.mutators.base import DistributionEstimationMutator


BASE_TEMPLATE = PromptTemplate.from_template(
    "INSTRUCTION GENOTYPES FOUND IN ASCENDING ORDER OF QUALITY"
    "\n{elites}\nINSTRUCTION: "
)
CHAT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a meta heuristic assisting in the development of "
            "better instructions to complete a task. Generate a new improved "
            "insutrction mutant to complete the task."
        ),
        HumanMessagePromptTemplate.from_template(
            "INSTRUCTION GENOTYPES FOUND IN ASCENDING ORDER OF QUALITY"
            "\n{elites}\nINSTRUCTION: "
        ),
    ]
)
PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=BASE_TEMPLATE,
    conditionals=[(is_chat_model, CHAT_TEMPLATE)],
)


# Lineage based mutation will only function if the elites are filled in.
# AddElites mutator must be active somewhere (likely in the post_step)


class LineageBasedMutation(LLMChain, DistributionEstimationMutator):
    """For each unit of evolution, we store a history of the individuals in its lineage
    that were the best in the population, i.e., a historical chronological list of
    elites. This list is provided to the LLM in chronological order (not filtered by
    diversity), with the heading "GENOTYPES FOUND IN ASCENDING ORDER OF QUALITY" to
    produce a novel prompt as continuation.
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
            prompt=PROMPT_SELECTOR.get_prompt(llm),
            mutation_prompt_factory=mutation_prompt_factory,
            task_prompt_factory=task_prompt_factory,
            **kwargs,
        )

    def prep_prompts(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Tuple[List[PromptValue], Optional[List[str]]]:
        stop = None
        if len(input_list) == 0:
            return [], stop

        for inputs in input_list:
            inputs["elites"] = "\n".join(
                ["INSTRUCTION: " + str(x) for x in inputs["elites"]]
            )

        return super().prep_prompts(input_list, run_manager)

    async def aprep_prompts(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Tuple[List[PromptValue], Optional[List[str]]]:
        stop = None
        if len(input_list) == 0:
            return [], stop

        for inputs in input_list:
            inputs["elites"] = "\n".join(
                ["INSTRUCTION: " + str(x) for x in inputs["elites"]]
            )

        return await super().aprep_prompts(input_list, run_manager)
