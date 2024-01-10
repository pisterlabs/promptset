from typing import List, Optional, Tuple

from pydantic import BaseModel, Field
from langchain import PromptTemplate, LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.output_parsers import PydanticOutputParser

from ai.question_answering.thought.base import (
    Thought,
    ThoughtPairComparison,
    ThoughtPairComparer,
)
from ai.question_answering.generators.thought.llm_scoring import (
    ScorerQuery,
    SCORE_QUERY_OUTPUT_PARSER,
)


class Response(BaseModel):
    response: Optional[List[str]]


OUTPUT_PARSER = PydanticOutputParser(pydantic_object=Response)


SHARED_PROMPT_TEMPLATE = """
You are a data analyst assessing 2 distinct data-conclusion thoughts provided. 
You are concerned with what the 2 thoughts share in common. 
You are to extract the shared information from the 2 thoughts.

Thought 1: {thought1}

Thought 2: {thought2}

{format_instructions}

Answer only with a list of the shared (in agreement) information between the two thoughts.
If there are no shared thoughts, answer with an empty list.

"""

UNIQUE_FROM_1_PROMPT_TEMPLATE = """
You are a data analyst assessing 2 distinct data-conclusion thoughts provided. 
You are concerned with what is unique to thought 1.
You are to extract the unique information from thought 1.

Thought 1: {thought1}

Thought 2: {thought2}

{format_instructions}

Answer only with a list of the unique information from thought 1.
If there are no unique thoughts from thought 1, answer with an empty list.

"""

UNIQUE_FROM_2_PROMPT_TEMPLATE = """
You are a data analyst assessing 2 distinct data-conclusion thoughts provided.
You are concerned with what is unique to thought 2.
You are to extract the unique information from thought 2.

Thought 1: {thought1}

Thought 2: {thought2}

{format_instructions}

Answer only with a list of the unique information from thought 2.
If there are no unique thoughts from thought 2, answer with an empty list.

"""


SHARED_PROMPT = PromptTemplate(
    template=SHARED_PROMPT_TEMPLATE,
    input_variables=["thought1", "thought2"],
    partial_variables={"format_instructions": OUTPUT_PARSER.get_format_instructions()},
    output_parser=OUTPUT_PARSER,
)

UNIQUE_FROM_1_PROMPT = PromptTemplate(
    template=UNIQUE_FROM_1_PROMPT_TEMPLATE,
    input_variables=["thought1", "thought2"],
    partial_variables={"format_instructions": OUTPUT_PARSER.get_format_instructions()},
    output_parser=OUTPUT_PARSER,
)

UNIQUE_FROM_2_PROMPT = PromptTemplate(
    template=UNIQUE_FROM_2_PROMPT_TEMPLATE,
    input_variables=["thought1", "thought2"],
    partial_variables={"format_instructions": OUTPUT_PARSER.get_format_instructions()},
    output_parser=OUTPUT_PARSER,
)


class ContradictionEvaluation(BaseModel):
    element_of_thought_1_in_contradiction: str
    element_of_thought_2_in_contradiction: str
    reason_elements_are_in_contradiction: str


class ConstradictionResponse(BaseModel):
    response: Optional[Tuple[ContradictionEvaluation, ...]] = Field(
        description="Contradictory thought tuples."
        # description="A list of tuples where the first element is the element of "
        # "thought1 that contradicts the second element (which comes from thought2). "
        # "The elements of each tuple must be directly contradictory of one another. "
        # "Not all parts of thoughts are contradictory."
    )


CONTRADITION_OUTPUT_PARSER = PydanticOutputParser(
    pydantic_object=ConstradictionResponse
)

CONTRADICTION_PROMPT_TEMPLATE = """
You are a data analyst assessing 2 distinct data-conclusion thoughts provided.
You are concerned with what is contradictory between the 2 thoughts.
You are to extract the contradictory information from the 2 thoughts.

Thought 1: {thought1}

Thought 2: {thought2}


Answer only with a list of the contradictory information between the two thoughts.
If there are no contradictory thoughts, answer with an empty list.
If a part of a thought doesnt contradict the other thought dont include it in the list of tuples.

You must only answer with discussions that are in direct contradiction.
Contradictions must disagree with each other.
You must explain why the thoughts are in contradiction.
You must reply in the exact schema specified.

{format_instructions}

"""


CONTRADICTION_PROMPT = PromptTemplate(
    template=CONTRADICTION_PROMPT_TEMPLATE,
    input_variables=["thought1", "thought2"],
    partial_variables={
        "format_instructions": CONTRADITION_OUTPUT_PARSER.get_format_instructions()
    },
    output_parser=CONTRADITION_OUTPUT_PARSER,
)


SUMMARIZE_PROMPT_TEMPLATE = """
You are a data analyst assessing 2 distinct data-conclusion thoughts provided.
You are concerned with forming conclusions from the 2 thoughts.
You will be provided with the shared, unique, and contradictory information between the 2 thoughts.
You are to summarize the information from the 2 thoughts.

Thought 1: {thought1}

Thought 2: {thought2}

Shared information: 
{shared}

Unique information from thought 1: 
{unique_from_1}

Unique information from thought 2: 
{unique_from_2}

Contradictory information: 
{contradictory}

{format_instructions}

Answer only with a summarised conclusion about the thoughts. Be specific and concise.

"""


class SummarizeResponse(BaseModel):
    response: str


SUMMARIZE_OUTPUT_PARSER = PydanticOutputParser(pydantic_object=SummarizeResponse)


SUMMARIZE_PROMPT = PromptTemplate(
    template=SUMMARIZE_PROMPT_TEMPLATE,
    input_variables=[
        "thought1",
        "thought2",
        "shared",
        "unique_from_1",
        "unique_from_2",
        "contradictory",
    ],
    partial_variables={
        "format_instructions": SUMMARIZE_OUTPUT_PARSER.get_format_instructions()
    },
    output_parser=SUMMARIZE_OUTPUT_PARSER,
)


SCORE_PROMPT_TEMPLATE = """
You are a data analyst assessing 2 distinct data-conclusion thoughts provided.
You are concerned with scoring conclusions fromed from the 2 thoughts.
You will be provided with the shared, unique, and contradictory information between the 2 thoughts.
You will also be provided with a conclusion formed from the 2 thoughts.
You are to score the conclusion on how well it summarizes the information from the 2 thoughts.
You are scoring on real scale from 0 to 1. 

Thought 1: {thought1}

Thought 2: {thought2}

Shared information: 
{shared}

Unique information from thought 1: 
{unique_from_1}

Unique information from thought 2: 
{unique_from_2}

Contradictory information: 
{contradictory}

Conclusions: 
{conclusion}

{format_instructions}

Answer only a score between 0 and 1. 
0.0 means the conclusion is completely wrong.
1.0 means the conclusion is completely correct.
0.5 means the conclusion is half correct.
< 0.5 means the conclusion is more wrong than right.
> 0.5 means the conclusion is more right than wrong.

"""

SCORE_PROMPT = PromptTemplate(
    template=SCORE_PROMPT_TEMPLATE,
    input_variables=[
        "thought1",
        "thought2",
        "shared",
        "unique_from_1",
        "unique_from_2",
        "contradictory",
        "conclusion",
    ],
    partial_variables={
        "format_instructions": SCORE_QUERY_OUTPUT_PARSER.get_format_instructions()
    },
    output_parser=SCORE_QUERY_OUTPUT_PARSER,
)


class LLThoughtPairComparer(ThoughtPairComparer):
    def __init__(
        self,
        llm: BaseLanguageModel,
    ):
        self.shared_chain = LLMChain(
            llm=llm,
            prompt=SHARED_PROMPT,
            output_parser=OUTPUT_PARSER,
        )
        self.unique_from_1_chain = LLMChain(
            llm=llm,
            prompt=UNIQUE_FROM_1_PROMPT,
            output_parser=OUTPUT_PARSER,
        )
        self.unique_from_2_chain = LLMChain(
            llm=llm,
            prompt=UNIQUE_FROM_2_PROMPT,
            output_parser=OUTPUT_PARSER,
        )
        self.contradition_chain = LLMChain(
            llm=llm,
            prompt=CONTRADICTION_PROMPT,
            output_parser=CONTRADITION_OUTPUT_PARSER,
        )
        self.discussion_chain = LLMChain(
            llm=llm,
            prompt=SUMMARIZE_PROMPT,
            output_parser=SUMMARIZE_OUTPUT_PARSER,
        )
        self.score_chain = LLMChain(
            llm=llm,
            prompt=SCORE_PROMPT,
            output_parser=SCORE_QUERY_OUTPUT_PARSER,
        )

    @staticmethod
    def _thought_repr(thought: Thought) -> str:
        return (
            "{ "
            + f"discussion: '{thought.discussion}', confidence: {thought.score}"
            + "}"
        )

    @staticmethod
    def _get_chain_response(
        self, chain, thought1: Thought, thought2: Thought
    ) -> List[str]:
        return chain.predict(
            **{
                "thought1": self._thought_repr(thought1),
                "thought2": self._thought_repr(thought2),
            }
        ).response

    def _get_shared(self, thought1: Thought, thought2: Thought) -> List[str]:
        return self._get_chain_response(self, self.shared_chain, thought1, thought2)

    def _get_unique_from_1(self, thought1: Thought, thought2: Thought) -> List[str]:
        return self._get_chain_response(
            self, self.unique_from_1_chain, thought1, thought2
        )

    def _get_unique_from_2(self, thought1: Thought, thought2: Thought) -> List[str]:
        return self._get_chain_response(
            self, self.unique_from_2_chain, thought1, thought2
        )

    def _get_contradictions(
        self, thought1: Thought, thought2: Thought
    ) -> List[Tuple[str, str]]:
        reply = self._get_chain_response(
            self, self.contradition_chain, thought1, thought2
        )
        if reply is None:
            return []
        return reply

    def _get_discussion(
        self,
        thought1: Thought,
        thought2: Thought,
        shared: List[str],
        unique: Tuple[List[str], List[str]],
        contraditions: List[Tuple[str, str]],
    ) -> str:
        return self.discussion_chain.predict(
            **{
                "thought1": self._thought_repr(thought1),
                "thought2": self._thought_repr(thought2),
                "shared": "\n".join(shared),
                "unique_from_1": "\n".join(unique[0]),
                "unique_from_2": "\n".join(unique[1]),
                "contradictory": "\n".join(
                    [
                        f"(i) {x.element_of_thought_1_in_contradiction} (ii) {x.element_of_thought_2_in_contradiction}"
                        for x in contraditions
                    ]
                ),
            }
        ).response

    def _get_score(
        self,
        thought1: Thought,
        thought2: Thought,
        shared: List[str],
        unique: Tuple[List[str], List[str]],
        contraditions: List[Tuple[str, str]],
        conclusion: str,
    ) -> float:
        return self.score_chain.predict(
            **{
                "thought1": str(thought1),
                "thought2": str(thought2),
                "shared": "\n".join(shared),
                "unique_from_1": "\n".join(unique[0]),
                "unique_from_2": "\n".join(unique[1]),
                "contradictory": "\n".join(
                    f"(i) {x.element_of_thought_1_in_contradiction} (ii) {x.element_of_thought_2_in_contradiction}"
                    for x in contraditions
                ),
                "conclusion": conclusion,
            }
        ).score
