from typing import List, Generic, TypeVar
from abc import ABC, abstractmethod

from langchain.embeddings.base import Embeddings

from ai.question_answering.schema import Hypothesis, Thought


T = TypeVar("T")


class EmbeddingSimmilarityThreshold(Generic[T], ABC):
    def __init__(self, embeddings_model: Embeddings, threshold: float = 0.8):
        self.embeddings_model = embeddings_model
        self.threshold = threshold

    @abstractmethod
    def _get_str_value(self, t: T) -> str:
        pass

    def __call__(self, t: T, other: List[T]) -> bool:
        others = [self._get_str_value(o) for o in other]
        return (
            self.embeddings_model.simmilarity(self._get_str_value(t), others)
            > self.threshold
        )


class HypothesisSimmilarityThreshold(EmbeddingSimmilarityThreshold[Hypothesis]):
    def _get_str_value(self, hypothesis: Hypothesis) -> str:
        return hypothesis.hypothesis


class ThoughtSimmilarityThreshold(EmbeddingSimmilarityThreshold[Thought]):
    def _get_str_value(self, thought: Thought) -> str:
        return thought.discussion
