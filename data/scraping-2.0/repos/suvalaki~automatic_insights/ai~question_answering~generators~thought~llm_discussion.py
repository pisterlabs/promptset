from abc import ABC, abstractclassmethod

from pydantic import BaseModel
from langchain import PromptTemplate, LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.output_parsers import PydanticOutputParser

from ai.question_answering.schema import Hypothesis, Thought


class DiscussionGenerator:
    def __call__(self, hypothesis: Hypothesis, data: str) -> str:
        return data


EXPLAINER_PROMPT_TEMPLATE = """

{format_instructions}

You are to scrutinize data.
Explain how the data is and how it relates to the hypothesis. 

Hypothesis: 
{hypothesis}

Data:
{data}

Explain how the data is and how it relates to the hypothesis. 
Dont make any claims about the truth of the hypothesis or not.
Dont talk about support or not of the hypothesis.


Answer only with the output schema specified.

"""


class ExplainerQuery(BaseModel):
    explanation: str


EXPLAINER_PARSER = PydanticOutputParser(pydantic_object=ExplainerQuery)


EXPLAINER_PROMPT = PromptTemplate(
    template=EXPLAINER_PROMPT_TEMPLATE,
    input_variables=["hypothesis", "data"],
    partial_variables={
        "format_instructions": EXPLAINER_PARSER.get_format_instructions()
    },
    output_parser=EXPLAINER_PARSER,
)


class LLMHypothesisDataExplainer(DiscussionGenerator):
    def __init__(self, llm):
        self.chain = LLMChain(
            llm=llm,
            prompt=EXPLAINER_PROMPT,
            output_parser=EXPLAINER_PARSER,
        )

    def __call__(self, hypothesis: Hypothesis, data: str) -> str:
        # return data
        # TODO:  reimplement
        result = self.chain.predict(
            hypothesis=hypothesis.hypothesis,
            data=data,
        )
        return result.explanation
