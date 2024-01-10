from typing import List, Dict, Any, Optional, Tuple, Callable, Type
import random

from langchain.llms.base import BaseLanguageModel
from langchain.embeddings.base import Embeddings
from langchain.chains.llm import LLMChain, PromptValue
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema.messages import SystemMessage
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.evaluation.embedding_distance.base import (
    EmbeddingDistance,
    EmbeddingDistanceEvalChain,
)
from langchain.callbacks.manager import (
    CallbackManagerForChainRun,
    AsyncCallbackManagerForChainRun,
)

from prompt_breeder.types import (
    TaskPrompt,
    MutationPrompt,
)
from prompt_breeder.mutators.base import DistributionEstimationMutator


BASE_TEMPLATE = PromptTemplate.from_template("{task_prompt_set}  INSTRUCTION MUTATNT: ")
CHAT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a meta heuristic assisting in the development of "
            "better instructions to complete a task. Generate a new improved "
            "insutrction mutant to complete the task."
        ),
        HumanMessagePromptTemplate.from_template(
            "{task_prompt_set}  INSTRUCTION MUTATNT: "
        ),
    ]
)
PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=BASE_TEMPLATE,
    conditionals=[(is_chat_model, CHAT_TEMPLATE)],
)


class EstimationOfDistributionMutation(LLMChain, DistributionEstimationMutator):
    """We generate the initial task-prompts by concatenating (for each task-
    prompt) a randomly drawn ‘mutation-prompt’ (e.g. "Make a variant of the prompt.")
    and a randomly drawn ‘thinking-style’ (e.g. "Let’s think step by step") to the
    problem description, and provide that to the LLM to produce a continuation,
    resulting in an initial task-prompt
    """

    embed_scorer: EmbeddingDistanceEvalChain
    threshold: float = 0.05  # distance between the keys

    @classmethod
    def from_llm(
        cls: Type["EstimationOfDistributionMutation"],
        task_prompt_factory: Callable[[str], TaskPrompt],
        mutation_prompt_factory: Callable[[str], MutationPrompt],
        llm: BaseLanguageModel,
        embeddings: Embeddings,
        distance_metric: EmbeddingDistance,
        **kwargs,
    ) -> "EstimationOfDistributionMutation":
        return cls(
            task_prompt_factory=task_prompt_factory,
            mutation_prompt_factory=mutation_prompt_factory,
            llm=llm,
            prompt=PROMPT_SELECTOR.get_prompt(llm),
            embed_scorer=EmbeddingDistanceEvalChain(
                embeddings=embeddings, distance_metric=distance_metric, **kwargs
            ),
            **kwargs,
        )

    @property
    def input_keys(self) -> List[str]:
        return self.prompt.input_variables

    def sort_population(
        self, task_prompt_set: List[TaskPrompt], run_manager=None, **kwargs
    ) -> None:
        # Random shuffle
        random.shuffle(task_prompt_set)

    def _validate_score_single(
        self, p0: TaskPrompt, p1: TaskPrompt, callbacks, **kwargs
    ):
        score = self.embed_scorer.evaluate_strings(
            prediction=str(p0), reference=str(p1), callbacks=callbacks, **kwargs
        )
        return abs(score["score"]) < self.threshold

    def filter_population(
        self, task_prompt_set: List[TaskPrompt], callbacks=None, **kwargs
    ) -> List[TaskPrompt]:
        # filter the population of prompts on the basis of
        # BERT (Devlin et al., 2019) embedding cosine similarities between each other—an
        # individual is not included in the list if it is more than 0.95 similar to
        # any other entry in the list,
        #
        # Here instead allow for ANY embedding model and any threshold
        #
        filtered_pop: List[TaskPrompt] = []
        for i, p0 in enumerate(task_prompt_set):
            if not any(
                [
                    self._validate_score_single(p0, p1, callbacks, **kwargs)
                    for p1 in filtered_pop
                ]
            ):
                filtered_pop += [p0]

        return filtered_pop

    def _singleton_prompt_prep(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[
            CallbackManagerForChainRun | AsyncCallbackManagerForChainRun
        ] = None,
    ) -> None:
        filtered_pop = self.filter_population(
            inputs["task_prompt_set"],
            run_manager.get_child() if run_manager else None,
        )
        self.sort_population(filtered_pop)
        filtered_pop_str = "  ".join(
            [f"INSTRUCTION: {str(task_prompt)}" for task_prompt in filtered_pop]
        )
        inputs["task_prompt_set"] = filtered_pop_str

    def prep_prompts(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Tuple[List[PromptValue], Optional[List[str]]]:
        stop = None

        if len(input_list) == 0:
            return [], stop

        # unpack the filtered population
        for inputs in input_list:
            self._singleton_prompt_prep(inputs, run_manager)

        return super().prep_prompts(input_list, run_manager)

    async def aprep_prompts(
        self,
        input_list: List[Dict[str, Any]],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Tuple[List[PromptValue], Optional[List[str]]]:
        stop = None

        if len(input_list) == 0:
            return [], stop

        # unpack the filtered population
        for inputs in input_list:
            self._singleton_prompt_prep(inputs, run_manager)

        return await super().aprep_prompts(input_list, run_manager)
