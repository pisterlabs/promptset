import asyncio
from typing import List, Callable, Type
from langchain.llms.base import BaseLanguageModel
from langchain.embeddings.base import Embeddings
from langchain.evaluation.embedding_distance.base import (
    EmbeddingDistance,
    EmbeddingDistanceEvalChain,
)
from langchain.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema.messages import SystemMessage
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from prompt_breeder.types import TaskPrompt, MutationPrompt
from prompt_breeder.evolution.fitness import FitnessScorer
from prompt_breeder.mutators.estimation_of_distribution_mutation import (
    EstimationOfDistributionMutation,
)


BASE_TEMPLATE = PromptTemplate.from_template(
    "INSTRUCTION: {mutation_prompt}"
    + "\n A List ofResponses in descending order of score. "
    + "{task_prompt_set}  INSTRUCTION MUTATNT: "
)
CHAT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        SystemMessage(
            content="You are a meta heuristic assisting in the development of "
            "better instructions to complete a task. Generate a new improved "
            "insutrction mutant to complete the task."
        ),
        HumanMessagePromptTemplate.from_template(
            "INSTRUCTION: {mutation_prompt}"
            + "\n A List ofResponses in descending order of score. "
            + "{task_prompt_set}  INSTRUCTION MUTATNT: "
        ),
    ]
)
PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=BASE_TEMPLATE,
    conditionals=[(is_chat_model, CHAT_TEMPLATE)],
)


class EdaRankAndIndexMutation(EstimationOfDistributionMutation):
    fitness_scorer: FitnessScorer

    @classmethod
    def from_llm(  # type: ignore[override]
        cls: Type["EdaRankAndIndexMutation"],
        task_prompt_factory: Callable[[str], TaskPrompt],
        mutation_prompt_factory: Callable[[str], MutationPrompt],
        llm: BaseLanguageModel,
        embeddings: Embeddings,
        distance_metric: EmbeddingDistance,
        fitness_scorer: FitnessScorer,
        **kwargs,
    ) -> "EdaRankAndIndexMutation":
        return cls(
            task_prompt_factory=task_prompt_factory,
            mutation_prompt_factory=mutation_prompt_factory,
            llm=llm,
            prompt=PROMPT_SELECTOR.get_prompt(llm),
            embed_scorer=EmbeddingDistanceEvalChain(
                embeddings=embeddings, distance_metric=distance_metric
            ),
            fitness_scorer=fitness_scorer,
            **kwargs,
        )

    def sort_population(
        self, task_prompt_set: List[TaskPrompt], run_manager=None, **kwargs
    ) -> None:
        # sort by fitness
        cb = run_manager.get_child() if run_manager else None
        fitnesses = [
            self.fitness_scorer.score(prompt, callbacks=cb)
            for prompt in task_prompt_set
        ]
        pairs = list(zip(task_prompt_set, fitnesses))
        pairs.sort(key=lambda x: x[1], reverse=True)
        task_prompt_set[:] = [pair[0] for pair in pairs]

    async def asort_population(
        self, task_prompt_set: List[TaskPrompt], run_manager=None, **kwargs
    ) -> None:
        # sort by fitness
        cb = run_manager.get_child() if run_manager else None
        fitnesses = await asyncio.gather(
            *[
                self.fitness_scorer.ascore(prompt, callbacks=cb)
                for prompt in task_prompt_set
            ]
        )
        pairs = list(zip(task_prompt_set, fitnesses))
        pairs.sort(key=lambda x: x[1], reverse=True)
        task_prompt_set[:] = [pair[0] for pair in pairs]
