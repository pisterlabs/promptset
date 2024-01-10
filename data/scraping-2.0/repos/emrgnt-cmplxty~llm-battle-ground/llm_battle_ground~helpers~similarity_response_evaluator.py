import numpy as np
from automata.llm import OpenAIEmbeddingProvider

from llm_battle_ground.helpers.leet_code_processor import LeetCodeProcessor


class SimilarityResponseEvaluator:
    """Class to evaluate LLM responses."""

    def __init__(
        self,
        embedding_provider: OpenAIEmbeddingProvider = OpenAIEmbeddingProvider(),
    ) -> None:
        self.embedding_provider = embedding_provider

    @staticmethod
    def evaluate_common_problems(dict1: dict, dict2: dict) -> dict:
        """Evaluate the common problems between two dictionaries."""
        return {
            key: value
            for key, value in dict2.items()
            if any(
                value["problem_number"] == item["problem_number"]
                for item in dict1.values()
            )
        }

    @staticmethod
    def unpack_to_string(data: dict) -> str:
        """Unpacks an appropriately formatted LeetCode problem dictionary into a string."""
        return "".join(
            f"Example {example_num}:\nLeetCode Problem #{details['problem_number']}\nTitle: {details['title']}\nDescription:\n{details['description']}\n\n"
            for example_num, details in data.items()
        )

    def clean_and_build_embedded_response(self, response: str) -> np.ndarray:
        """Helper function to clean and build embedded response."""
        cleaned_response = LeetCodeProcessor.clean_response(response)
        return self.embedding_provider.build_embedding_vector(cleaned_response)
