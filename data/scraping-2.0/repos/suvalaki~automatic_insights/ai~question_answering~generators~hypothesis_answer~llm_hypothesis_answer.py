from typing import Callable, List, TypeVar, Generic

from pydantic import BaseModel
from langchain import PromptTemplate, LLMChain
from langchain.chains import TransformChain, SequentialChain
from langchain.base_language import BaseLanguageModel
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.output_parsers import PydanticOutputParser

from ai.question_answering.schema import Thought, Hypothesis, AnsweredHypothesis
from ai.question_answering.generators.base import (
    TargettedThoughtGenerator,
    ThoughtSummarizer,
    HypothesisAnswerGenerator,
)

HYPOTHESIS_ANSWER_CONCLUSION_PROMPT_TEMPLATE = """
You are to act as a data insights analyst. 
You will be given a hypothesis and discussions based on one or more data extracts.
The data relates to the hypothesis.
The discussions are produced in an effort to provide evidence to answer the hypothesis.
You are to form conclusions about the hypothesis based on the discussion provided.
You must comment on if the discussion actually anwers the question. If it does not say so.
It might be the case that there is uncertainty that the discussion provides enough information to answer the hypothesis. If this is the case you should note it.

Hypothesis: 
{hypothesis}

Discussions:
{discussions}


You should be factual in your response. 
You must refer to the discussion evidence specifically in your discussion.
If any of the thoughts or combinations of the thoughts answers the hypothesis you should say so.
You must have enough information to completely support the hypothesis if you suggest it is supported.
You must not make additional assumptions than the data provided.
Explain what additional data is needed if it isnt answered.

You are to answer if enough data has been supplied to answer the hypothesis.

{format_instructions}

"""


class HypothesisAnswerConclusion(BaseModel):
    conclusion: str


HYPOTHESIS_ANSWER_CONCLUSION_OUTPUT_PARSER = PydanticOutputParser(
    pydantic_object=HypothesisAnswerConclusion
)


HYPOTHESIS_ANSWER_CONCLUSION_PROMPT = PromptTemplate(
    template=HYPOTHESIS_ANSWER_CONCLUSION_PROMPT_TEMPLATE,
    input_variables=["hypothesis", "discussions"],
    partial_variables={
        "format_instructions": HYPOTHESIS_ANSWER_CONCLUSION_OUTPUT_PARSER.get_format_instructions()
    },
    output_parser=HYPOTHESIS_ANSWER_CONCLUSION_OUTPUT_PARSER,
)


HYPOTHESIS_ANSWER_SCORE_PROMPT_TEMPLATE = """
You are to act as a data insights analyst. 
You will be given a hypothesis and discussions based on one or more data extracts.
You will be given a conclusion based on the input discussions.
The data relates to the hypothesis.
The discussions are produced in an effort to provide evidence to answer the hypothesis.
You are to score how likely the hypothesis is to be answered based on the inputs.

Hypothesis: 
{hypothesis}

Discussions:
{discussions}

Conclusion to evaluate:
{conclusion}


{format_instructions}

Answer only a score between 0 and 1. 
0.0 means there is no information available to confirm or deny the hypothesis.
1.0 means there is complete available to confirm or deny the hypothesis.
0.5 means there is 50% certainty that we are able to confirm or deny the hypothesis.
0.2 means there is 20% certainty that we are able to confirm or deny the hypothesis.
0.8 means there is 80% certainty that we are able to confirm or deny the hypothesis.

"""


class HypothesisAnswerScore(BaseModel):
    score: float


HYPOTHESIS_ANSWER_SCORE_OUTPUT_PARSER = PydanticOutputParser(
    pydantic_object=HypothesisAnswerScore
)


HYPOTHESIS_ANSWER_SCORE_PROMPT = PromptTemplate(
    template=HYPOTHESIS_ANSWER_SCORE_PROMPT_TEMPLATE,
    input_variables=["hypothesis", "discussions", "conclusion"],
    partial_variables={
        "format_instructions": HYPOTHESIS_ANSWER_SCORE_OUTPUT_PARSER.get_format_instructions()
    },
    output_parser=HYPOTHESIS_ANSWER_SCORE_OUTPUT_PARSER,
)


class LLMHypothesisEvaluator:
    def __init__(
        self,
        llm: BaseLanguageModel,
    ):
        self._conclusion_chain = LLMChain(
            llm=llm,
            prompt=HYPOTHESIS_ANSWER_CONCLUSION_PROMPT,
            output_parser=HYPOTHESIS_ANSWER_CONCLUSION_PROMPT.output_parser,
            output_key="conclusion",
        )
        self._scoring_prepare_transform = TransformChain(
            input_variables=["conclusion"],
            output_variables=["conclusion_str"],
            transform=lambda x: {"conclusion_str": x["conclusion"].conclusion},
        )
        self._scoring_chain = LLMChain(
            llm=llm,
            prompt=HYPOTHESIS_ANSWER_SCORE_PROMPT,
            output_parser=HYPOTHESIS_ANSWER_SCORE_PROMPT.output_parser,
            output_key="score",
        )
        self._parse_to_thought = TransformChain(
            input_variables=["conclusion_str", "score"],
            output_variables=["thought"],
            transform=lambda x: {
                "thought": Thought(
                    discussion=x["conclusion_str"], score=x["score"].score
                )
            },
        )
        self._chain = SequentialChain(
            chains=[
                self._conclusion_chain,
                self._scoring_prepare_transform,
                self._scoring_chain,
                self._parse_to_thought,
            ],
            input_variables=["hypothesis", "discussions"],
            return_all=True,
        )

    def __call__(
        self, hypothesis: Hypothesis, thoughts: List[Thought], answered: Thought
    ):
        return self._chain(
            dict(hypothesis=hypothesis.hypothesis, discussions=answered.discussion)
        )["thought"]
