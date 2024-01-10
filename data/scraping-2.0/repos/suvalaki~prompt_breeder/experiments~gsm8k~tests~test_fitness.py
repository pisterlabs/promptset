import pytest  # noqa: F401
from typing import List, Optional, Any, Mapping
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from prompt_breeder.evolution.fitness import BestMemberFitness
from prompt_breeder.prompts.string import StringTaskPrompt
from experiments.gsm8k.fitness import NiaveContainsCorrectAnswer

dataset = [
    # TODO: replace this with some dataset ingestion method
    # probably use huggingface datasets
    {"question": "What is 3+ 9?", "answer": "12"},
    {"question": "What is 13+ 5?", "answer": "18"},
]


class MockLLM(LLM):
    """Hijack LLM so that it works with regex parser.
    Need to have admissible inputs"""

    values: str

    @property
    def _llm_type(self) -> str:
        return "custom"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        if stop is not None:
            raise ValueError("stop kwargs are not permitted.")
        return self.values

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}


# Since 1 will be correct from the 2 data points. Fitness should be 0.5


def test_runs():
    llm = MockLLM(values="12")
    scorer = NiaveContainsCorrectAnswer.from_llm(llm=llm, dataset=dataset, verbose=1)
    prompt = StringTaskPrompt(text="Solve the math word problem, show your workings.")
    score = scorer.score(prompt)
    assert score == 0.5


def test_runs_with_zero_score():
    llm = MockLLM(values="00")
    scorer = NiaveContainsCorrectAnswer.from_llm(llm=llm, dataset=dataset, verbose=1)
    prompt = StringTaskPrompt(text="Solve the math word problem, show your workings.")
    score = scorer.score(prompt)
    assert score == 0.0


def test_max_fitness_over_pop():
    llm = MockLLM(values="12")
    scorer = NiaveContainsCorrectAnswer.from_llm(
        llm=llm, dataset=dataset, data_aggfunc=lambda a, b: sum(b), verbose=1
    )
    multi_scorer = BestMemberFitness(scorer=scorer)
    prompt0 = StringTaskPrompt(text="Solve the math word problem, show your workings.")
    prompt1 = StringTaskPrompt(text="Solve the math word problem.")
    score = multi_scorer.score([prompt0, prompt1])
    # Max fitness inside the list is 1.0
    assert score == 1.0
