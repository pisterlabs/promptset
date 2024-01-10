import pytest  # noqa: F401
import asyncio
from typing import Dict, List, Any, Optional
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.evaluation import load_evaluator
from langchain.evaluation.embedding_distance.base import (
    EmbeddingDistance,
)

from prompt_breeder.types import UnitOfEvolution, Population
from prompt_breeder.prompts.string import (
    StringPrompt,
    StringTaskPrompt,
    StringMutationPrompt,
)
from prompt_breeder.mutators.estimation_of_distribution_mutation import (
    EstimationOfDistributionMutation,
)


KEY0 = "key0"
KEY1 = "key1"

# Hardcode the embeddings with known cosine distances between
# them.
FIXED_EMBEDDINGS: Dict[str, List[float]] = {
    KEY0: [1.0, 1.0, 1.0, 1.0],
    KEY1: [-1.0, -1.0, -1.0, -1.0],
}


class MockFixedEmbeddings(Embeddings):
    """Embeddings that always return the fixed membership we defined
    within FIXED_EMBEDDINGS"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return FIXED_EMBEDDINGS[text]


# "{task_prompt_set}  INSTRUCTION MUTATNT: "
FIXED_PROMPT_REPLY: Dict[str, str] = {
    KEY0: "",
    KEY1: "",
}


class MockLLM(LLM):
    """Hijack LLM so that it works with regex parser.
    Need to have admissible inputs"""

    @property
    def _llm_type(self) -> str:
        return "custom_estimation_of_distribution"

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


def test_same_string_is_filtered():
    llm = MockLLM()
    embed_model = MockFixedEmbeddings()
    prompt0 = StringTaskPrompt(text="key0")
    prompt1 = StringTaskPrompt(text="key0")
    unit = UnitOfEvolution(  # noqa: F841
        problem_description=StringPrompt(text="ignored by ED mutation"),
        task_prompt_set=[
            prompt0,
            prompt1,
        ],
        mutation_prompt=StringMutationPrompt(text="ignored by ED mutation"),
        elites=[],
    )
    scorer = load_evaluator(
        "embedding_distance",
        distance_metric=EmbeddingDistance.EUCLIDEAN,
        embeddings=embed_model,
    )
    _ = scorer.evaluate_strings(
        prediction=str(prompt0),
        reference=str(prompt1),
    )
    mutator = EstimationOfDistributionMutation.from_llm(
        llm=llm,
        embeddings=embed_model,
        distance_metric=EmbeddingDistance.EUCLIDEAN,
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        verbose=1,
    )
    ans = mutator.filter_population([prompt0, prompt1])
    assert len(ans) == 1


def test_different_string_is_not_filtered():
    llm = MockLLM()
    embed_model = MockFixedEmbeddings()
    prompt0 = StringTaskPrompt(text="key0")
    prompt1 = StringTaskPrompt(text="key1")
    unit = UnitOfEvolution(  # noqa: F841
        problem_description=StringPrompt(text="ignored by ED mutation"),
        task_prompt_set=[
            prompt0,
            prompt1,
        ],
        mutation_prompt=StringMutationPrompt(text="ignored by ED mutation"),
        elites=[],
    )
    scorer = load_evaluator(
        "embedding_distance",
        distance_metric=EmbeddingDistance.EUCLIDEAN,
        embeddings=embed_model,
    )
    _ = scorer.evaluate_strings(
        prediction=str(prompt0),
        reference=str(prompt1),
    )
    mutator = EstimationOfDistributionMutation.from_llm(
        llm=llm,
        embeddings=embed_model,
        distance_metric=EmbeddingDistance.EUCLIDEAN,
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        verbose=1,
    )
    ans = mutator.filter_population([prompt0, prompt1])
    assert len(ans) == 2


def test_runs_over_unit():
    llm = MockLLM()
    embed_model = MockFixedEmbeddings()
    prompt0 = StringTaskPrompt(text="key0")
    prompt1 = StringTaskPrompt(text="key0")
    unit = UnitOfEvolution(  # noqa: F841
        problem_description=StringPrompt(text="ignored by ED mutation"),
        task_prompt_set=[
            prompt0,
            prompt1,
        ],
        mutation_prompt=StringMutationPrompt(text="ignored by ED mutation"),
        elites=[],
    )
    scorer = load_evaluator(
        "embedding_distance",
        distance_metric=EmbeddingDistance.COSINE,
        embeddings=embed_model,
    )
    _ = scorer._evaluate_strings(
        prediction=str(prompt0),
        reference=str(prompt1),
    )
    mutator = EstimationOfDistributionMutation.from_llm(
        llm=llm,
        embeddings=embed_model,
        distance_metric=EmbeddingDistance.COSINE,
        task_prompt_factory=lambda x: StringTaskPrompt(text=x),
        mutation_prompt_factory=lambda x: StringMutationPrompt(text=x),
        verbose=1,
    )
    population = Population(members=[unit])
    _ = mutator.mutate(population, unit)
    _ = asyncio.run(mutator.amutate(population, unit))
