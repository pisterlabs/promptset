from abc import ABC, abstractmethod

import openai
from openai.embeddings_utils import distances_from_embeddings

from openaiapp.embeddings import AbstractEmbeddings
from openaiapp.text_preparators import AbstractTextPreparatory


class AbstractAIQuestionAnswering(ABC):
    """
    Abstract base class for AI-based question answering systems.
    """

    @abstractmethod
    def create_context(self, question: str) -> str:
        """
        Create a context for a question by finding the most similar context from a data frame.
        """
        pass

    @abstractmethod
    def answer_question(self, question: str) -> str:
        """
        Answer a question based on a context derived from a data frame.
        """
        pass


class AIQuestionAnsweringBasedOnContext(AbstractAIQuestionAnswering):
    """
    Implementation of AI-based question answering using context derived from a data frame.
    """

    def __init__(
        self,
        text_preparatory: AbstractTextPreparatory,
        text_embeddings_object: AbstractEmbeddings,
        model: str,
        max_tokens: int,
        context_max_len: int,
        stop_sequence: str,
    ):
        """
        Initialize the AIQuestionAnsweringBasedOnContext object.
        """
        self.text_embeddings_object = text_embeddings_object
        self.text_preparatory = text_preparatory
        self.model = model
        self.max_tokens = max_tokens
        self._context_max_len = context_max_len
        self.stop_sequence = stop_sequence

    def create_context(self, question: str) -> str:
        """
        Create a context for a question by finding the most similar context from the data frame.
        """
        df_prepared = self.text_preparatory.generate_tokens_amount()
        q_embeddings = self.text_embeddings_object.create_embeddings(input=question)
        df_prepared["distances"] = distances_from_embeddings(
            q_embeddings, df_prepared["embeddings"].values, distance_metric="cosine"
        )

        context_texts = []
        current_length = 0

        for _, row in df_prepared.sort_values("distances", ascending=True).iterrows():
            current_length += row["n_tokens"]
            if current_length > self._context_max_len:
                break
            context_texts.append(row["text"])

        return "\n\n###\n\n".join(context_texts)

    def answer_question(self, question: str) -> str:
        """
        Answer a question based on the most similar context derived from the data frame.
        """
        context = self.create_context(question)
        try:
            response = openai.Completion.create(
                prompt=f"Context: {context}\n\nQuestion: {question}\nAnswer:",
                temperature=0,
                max_tokens=self.max_tokens,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stop=self.stop_sequence,
                model=self.model,
            )
            return response["choices"][0]["text"].strip()
        except openai.error.OpenAIError as e:
            raise RuntimeError(f"Error in generating answer from OpenAI: {e}.")
        except Exception as e:
            raise RuntimeError(f"Unexpected error in answer generation: {e}.")

    @property
    def context_max_len(self) -> int:
        """
        Get the maximum length of the context.
        """
        return self._context_max_len

    @context_max_len.setter
    def context_max_len(self, length: int):
        """
        Set the maximum length of the context.
        """
        if length < 0:
            raise ValueError("Context max length must be non-negative.")
        self._context_max_len = length
