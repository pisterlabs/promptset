from abc import ABC, abstractclassmethod

from pydantic import BaseModel
from langchain import PromptTemplate, LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.output_parsers import PydanticOutputParser

from ai.question_answering.schema import Hypothesis, Thought


class DiscussionGenerator(ABC):
    @abstractclassmethod
    def __call__(self, hypothesis: Hypothesis, data: str, conclustion: str) -> float:
        ...


SCORER_PROMPT_TEMPLATE = """
You are to act as a data insights analyst. 
You will be given a hypothesis, a piece of data that has been evaluated to relate to the hypothesis, and a conclusion that has been reached about the hypothesis.
You are to score between 0.0 and 1.0 how well the conclusion is supported by the data.

Hypothesis: 
{hypothesis}

Data:
{data}

Conclusion:
{conclusion}

{format_instructions}

You must only reply with a score between 0.0 and 1.0.
0.0 means that the conclusion is not supported by the data.
0.2 means that the conclusion is 20\% supported by the data.
0.5 means that the conclusion is neither supported nor unsupported by the data.
0.75 means that the conclusion is 75\% supported by the data.
1.0 means that the conclusion is supported by the data.


"""


class ScorerQuery(BaseModel):
    score: float


SCORE_QUERY_OUTPUT_PARSER = PydanticOutputParser(pydantic_object=ScorerQuery)


SCORER_PROMPT = PromptTemplate(
    template=SCORER_PROMPT_TEMPLATE,
    input_variables=["hypothesis", "data", "conclusion"],
    partial_variables={
        "format_instructions": SCORE_QUERY_OUTPUT_PARSER.get_format_instructions()
    },
    output_parser=SCORE_QUERY_OUTPUT_PARSER,
)


class LLMDataExplainerScorer(DiscussionGenerator):
    def __init__(self, llm):
        self.chain = LLMChain(
            llm=llm,
            prompt=SCORER_PROMPT,
            output_parser=SCORE_QUERY_OUTPUT_PARSER,
        )

    def __call__(self, hypothesis: Hypothesis, data: str, conclusion: str) -> str:
        return self.chain.predict(
            hypothesis=hypothesis,
            data=data,
            conclusion=conclusion,
        ).score
