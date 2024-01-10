from typing import Callable, List, TypeVar, Generic

from pydantic import BaseModel, Field
from langchain import PromptTemplate, LLMChain
from langchain.chains import TransformChain, SequentialChain
from langchain.base_language import BaseLanguageModel
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.output_parsers import PydanticOutputParser

from ai.question_answering.schema import Thought, Hypothesis, AnsweredHypothesis
from ai.question_answering.generators.base import (
    TargettedThoughtGenerator,
    ThoughtSummarizer,
    HypothesisConcluder,
    HypothesisAnswerGenerator,
)


# This is a chain of thought
class Ans(BaseModel):
    how_to_answer: List[str] = Field(
        description="everything required to answer the question in order."
    )
    necessary_data: List[str]
    detailed_complete_workings: List[str]
    final_calculation: str
    conclusion: str


class AnsThought(Thought):
    answer: Ans


HYPOTHESIS_CONCLUSION_PROMPT_TEMPLATE = """
You will be given a hypothesis and thoughts based on one or more data extracts.

Hypothesis: 
{hypothesis}


thoughts:
{thoughts}


Use the data provided to answer the hypothesis truthfully. 
You should use data that supports the hypothesis only.
Answer the hypothesis. You must provide data (from the thoughts) as evidence for your conclusion.
Show your complete and exhaistive calculations as an ordered list. Show the results of calculations.


{format_instructions}

Answer with your workings followed by your conclusions.
"""


HYPOTHESIS_CONCLUSION_OUTPUT_PARSER = PydanticOutputParser(pydantic_object=Ans)


HYPOTHESIS_CONCLUSION_PROMPT = PromptTemplate(
    template=HYPOTHESIS_CONCLUSION_PROMPT_TEMPLATE,
    input_variables=["hypothesis", "thoughts"],
    partial_variables={
        "format_instructions": HYPOTHESIS_CONCLUSION_OUTPUT_PARSER.get_format_instructions()
    },
    output_parser=HYPOTHESIS_CONCLUSION_OUTPUT_PARSER,
)


class LLMHypothesisConcluder(HypothesisConcluder):
    def __init__(self, llm: BaseLanguageModel):
        self._chain = LLMChain(
            llm=llm,
            prompt=HYPOTHESIS_CONCLUSION_PROMPT,
            output_parser=HYPOTHESIS_CONCLUSION_PROMPT.output_parser,
        )

    def __call__(self, h: Hypothesis, thoughts: List[Thought]) -> Thought:
        formatted_thoughts = "\n".join(f"{ {str(x)} }" for x in thoughts)
        reply = self._chain.predict(
            hypothesis=h.hypothesis, thoughts=formatted_thoughts
        )

        format_reply = f"""
```
workings: 
{reply.detailed_complete_workings}

calculation: 
{reply.final_calculation}

conclusion:
{reply.conclusion}
```

"""

        # format_reply = "\n".join(
        #     f"{key}: {value}"
        #     for key, value in reply.dict().items()
        #     if key in ["workings", "final_calculation", "conclusion"]
        # )
        return AnsThought(discussion=format_reply, score=0.0, answer=reply)
