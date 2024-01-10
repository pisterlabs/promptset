from abc import ABC, abstractmethod
import os
from typing import List
from langchain.llms.base import BaseLLM
from langchain import PromptTemplate
from langchain.chains.openai_functions import create_structured_output_chain

from core.code_review.models import (
    MethodInfo,
    MethodInfoCollection,
    ReviewCommentsCollection,
)


class BaseReviewCommentProvider(ABC):
    """Base class for review comment providers"""

    @abstractmethod
    def get_review_comments(
        self, code_file_contents: str, changed_methods: MethodInfoCollection, lang: str
    ) -> List[tuple[MethodInfo, ReviewCommentsCollection]]:
        """Returns a list of review comments for the given collection of changed methods"""
        raise NotImplementedError()


class ReviewCommentProvider(BaseReviewCommentProvider):
    """Provides review comments for a given collection of changed methods"""

    def __init__(self, llm: BaseLLM, verbose: bool = False) -> None:
        self.verbose = verbose
        self.llm = llm

    def get_review_comments(
        self, code_file_contents: str, changed_methods: MethodInfoCollection, lang: str
    ) -> List[tuple[MethodInfo, ReviewCommentsCollection]]:
        script_path = os.path.abspath(__file__)
        parent_directory = os.path.dirname(script_path)
        prompt_abs_path = os.path.join(parent_directory, "prompts", "review.txt")

        prompt = PromptTemplate.from_file(prompt_abs_path, ["input", "lang"])
        chain = create_structured_output_chain(
            ReviewCommentsCollection, llm=self.llm, prompt=prompt, verbose=self.verbose
        )
        results = []
        code_lines = code_file_contents.splitlines()
        for method in changed_methods.items:
            method_code = "\n".join(
                code_lines[method.start_line - 1 : method.end_line + 1]
            )
            result_dict = chain.run({"input": method_code, "lang": lang})
            review_comments = ReviewCommentsCollection.parse_obj(result_dict)
            results.append((method, review_comments))
        return results
