#  Blackboard-PAGI - LLM Proto-AGI using the Blackboard Pattern
#  Copyright (c) 2023. Andreas Kirsch
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU Affero General Public License as published
#  by the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU Affero General Public License for more details.
#
#  You should have received a copy of the GNU Affero General Public License
#  along with this program.  If not, see <https://www.gnu.org/licenses/>.

from dataclasses import dataclass
from typing import ClassVar, Generic, Tuple, TypeGuard, TypeVar

from langchain.llms import BaseLLM

from blackboard_pagi.prompts.chat_chain import ChatChain

T = TypeVar('T')


@dataclass
class LLMOptional:
    """
    A class to represent a potentially parsed value.
    """

    def is_missing(self) -> TypeGuard["LLMNone"]:
        raise NotImplementedError()


@dataclass
class LLMValue(LLMOptional, Generic[T]):
    """
    A class to represent a potentially parsed value.
    """

    value: T
    source: str

    def is_missing(self):
        return False


@dataclass
class LLMNone(LLMOptional):
    """
    A class to represent a missing value (failed conversion).
    """

    details: str

    def is_missing(self):
        return True


class ConversionFailure(LLMNone):
    pass


@dataclass
class LLMBool(LLMValue[bool]):
    def __bool__(self):
        return self.value


V = TypeVar("V", bound=LLMOptional)

L = TypeVar("L", bound=LLMValue)


@dataclass
class StructuredConverter(Generic[L]):
    query: ClassVar[str]
    """Follow-up query to use in prompt if the conversion fails."""
    examples: ClassVar[list[Tuple[str, str]]]
    """Examples to use in prompt with the fallback LLM if the direct conversion fails."""

    conversion_llm: BaseLLM | None = None
    """LLM to use for conversion (we usually try the chat model first though)."""

    def __call__(self, response: str) -> L | LLMNone:
        raise NotImplementedError()

    @staticmethod
    def strip_explanation(response):
        """
        Strips the explanation from a response.
        """
        if "## Explanation" in response:
            return response.split("## Explanation")[0].strip()
        return response

    def convert_from_chain(self, chat_chain: "ChatChain") -> L | LLMNone:
        """
        Converts a chat chain's last response to a value using the given converter.
        """
        result = self(self.strip_explanation(chat_chain.response))
        if not result.is_missing():
            return result

        query_prompt = (
            self.query
            + "\n\nExamples:\n"
            + "\n".join(f"INPUT:\n{example[0]}\n\nOUTPUT:\n{example[1]}\n" for example in self.examples)
        )

        query_response, chat_chain = chat_chain.query(query_prompt)
        result = self(self.strip_explanation(query_response))
        if not result.is_missing():
            return result

        for _ in range(2):
            assert isinstance(result, LLMNone)

            retry_prompt = (
                "I've failed at parsing your last answer. It did not follow the specification. Error:\n"
                f"{result.details}\n\n{self.query}.\n"
                "Do not apologize. Just follow instructions."
            )
            response, chat_chain = chat_chain.query(retry_prompt)
            result = self(self.strip_explanation(response))
            if not result.is_missing():
                return result

        if self.conversion_llm is not None:
            return self.convert_via_llm(query_response)

        return result

    def convert_via_llm(self, text: str) -> L | LLMNone:
        """
        Converts a text to a value using the given converter and conversion LLM.
        """
        assert self.conversion_llm is not None

        result = self(text)
        if not result.is_missing():
            return result

        # Build prompt for conversion LLM
        prompt = (
            f"{self.query}\n"
            "\n"
            "Examples:\n"
            "\n" + "\n".join(f"INPUT:\n{example[0]}\n\nOUTPUT:\n{example[1]}\n" for example in self.examples) + "\n"
            "INPUT:\n"
            f"{text}\n"
            "\n"
            "OUTPUT:\n"
        )

        result_text = self.conversion_llm(prompt, stop=["INPUT:"])
        result = self(result_text)

        if not result.is_missing():
            return result

        for _ in range(2):
            assert isinstance(result, LLMNone)

            retry_prompt = prompt + (
                "\n---\n"
                "I couldn't parse the last output and I failed.\n"
                "Error:\n"
                f"{result.details}\n"
                "\n"
                f"{self.query}\n"
                "Try again:\n"
                "---\n"
                "OUTPUT:\n"
            )
            result_text = self.conversion_llm(retry_prompt, stop=["INPUT:"])
            result = self(result_text)
            if not result.is_missing():
                return result

        return result


class BooleanConverter(StructuredConverter[LLMBool]):
    query: ClassVar[str] = """Does this mean yes or no? Please answer with either 'yes' or 'no' as your full answer."""
    examples = [("Yes, indeed.", "Yes"), ("No, not at all.", "No"), ("Positive. The statement is true.", "Yes")]
    no_synonyms: ClassVar[set[str]] = {"no", "false", "0"}
    yes_synonyms: ClassVar[set[str]] = {"yes", "true", "1"}

    @classmethod
    def __call__(cls, response: str) -> LLMBool | LLMNone:
        reduced_response = response.lower().strip().rstrip('.')
        if reduced_response in cls.no_synonyms | cls.yes_synonyms:
            return LLMBool(any(synonym in response.lower() for synonym in cls.yes_synonyms), response)
        else:
            return LLMNone(f"Expected response in `{cls.no_synonyms}`" " | `{self.yes_synonyms}`!")

    def convert_from_chain(self, chat_chain: "ChatChain") -> LLMBool | LLMNone:
        return super().convert_from_chain(chat_chain)


class StringConverter(StructuredConverter[LLMValue[str]]):
    query: ClassVar[str] = (
        "Wrapped the relevant part in \"\" at the start, followed by an optional "
        "explanation using the format '##Explanation\n{your_explanation}'."
    )
    examples = [
        ("The answer is \"42\".", "\"42\n##Explanation\nMy explanation for this can follow here.\""),
        ("The answer is \"Hello\\\nWorld!\".", "\"Hello\\nWorld!\"\n## Explanation\nMy explanation can follow here."),
    ]

    def __call__(self, response: str) -> LLMValue[str] | LLMNone:
        if response.strip().startswith('"') and response.strip().endswith('"'):
            return LLMValue(response.strip()[1:-1], response)
        else:
            return LLMNone(
                "Expected only the response wrapped in \"\" at the start of your reply, "
                "followed by an optional explanation in a subsection ## Explanation!"
            )


class ProbabilityConverter(StructuredConverter[LLMValue[float]]):
    query: ClassVar[
        str
    ] = "Please repeat only the relevant probability as a number between 0 and 1 as your full answer."
    examples = [
        ("The probability is 0.42.", "0.42\n## Explanation\nMy explanation can follow here."),
        ("The probability is 0.5.", "0.5\n## Explanation\nMy explanation can follow here."),
    ]

    def __call__(self, response: str) -> LLMValue[float] | LLMNone:
        # parse the response as a float
        try:
            probability = float(response.strip())
        except ValueError:
            return LLMNone(
                "Expected response to be a number, followed by "
                "an optional explanation in a subsection ## Explanation!"
            )

        # check that the probability is between 0 and 1
        if 0 <= probability <= 1:
            return LLMValue(probability, response)
        else:
            return LLMNone(
                "Expected response to be between 0 and 1, followed by "
                "an optional explanation in a subsection ## Explanation!"
            )
