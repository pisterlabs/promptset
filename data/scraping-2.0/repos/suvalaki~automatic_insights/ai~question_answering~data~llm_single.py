from typing import List
from abc import ABC, abstractmethod

from pydantic import BaseModel
from langchain import PromptTemplate, LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.output_parsers import PydanticOutputParser

from ai.question_answering.schema import Thought, Hypothesis, DataSourceSelection
from ai.question_answering.data.base import DataSourceSelector


DATASOURCE_SELECTOR_PROMPT_TEMPLATE = """
You are tasked with evaluating a set of data_sources (and data getting functions) which could be used to answer a hypothesis.
It is your goal to select which source to use, and to explain why you selected it. 
Your selection can be part of a multiple step strategy where you combine multiple data sources to calculate an answer to the hypothesis.
You may also evaluate if the hypothesis is not answerable with any data_source (or combinations of them) provided. 


The hypothesis is: 
{hypothesis}


The data_sources are:
{data_sources}



Select the whole name of the data_source.
If the hypothesis is not answerable with any data_source, you should reply with "None" as the data_source instead.


The set of allowable names are: 
{allowed_names}


You should take account of the data you already have so that you dont select the same data. 
Be exploratory in your selection.
It is not enough to be relevant to the hypothesis. The selection must actually be useful in answering the hypothesis too.
Factor in data which is already selected in the past based on the thoughts. You dont need data you already have.



Prior explored data_source discussions:
{prior}
Only select a data source that provides new information.


{format_instructions}
Answer only with one selected data_source. The name must be exactly as is provided in this input.
You must only reply with the name of one data_source or "None".

"""


class DatasourceSelectorQuery(BaseModel):
    what_is_required_to_answer_the_question: str
    data_source: str
    reason: str


OUTPUT_PARSER = PydanticOutputParser(pydantic_object=DatasourceSelectorQuery)


DATASOURCE_SELECTOR_PROMPT = PromptTemplate(
    template=DATASOURCE_SELECTOR_PROMPT_TEMPLATE,
    input_variables=["hypothesis", "data_sources", "prior", "allowed_names"],
    partial_variables={"format_instructions": OUTPUT_PARSER.get_format_instructions()},
    output_parser=OUTPUT_PARSER,
)


class LLMDataSourceSelector(DataSourceSelector):
    def __init__(self, data_sources: List[Tool], llm: BaseLanguageModel, **kwargs):
        super().__init__(data_sources)
        self.chain = LLMChain(
            llm=llm,
            prompt=DATASOURCE_SELECTOR_PROMPT,
            output_parser=OUTPUT_PARSER,
        )

    def __call__(
        self, hypothesis: Hypothesis, prior: List[Thought]
    ) -> DataSourceSelection | None:
        data_sources = hypothesis.data_sources
        reply = self.chain.predict(
            **{
                "hypothesis": hypothesis.hypothesis,
                "data_sources": "\n".join(
                    [
                        "{"
                        + f'name: "{data_source.name}" description: "{data_source.description}"'
                        + "}"
                        for data_source in data_sources
                    ]
                ),
                "prior": "\n".join(
                    [
                        "thought: {discussion: "
                        + t.discussion
                        + " score: "
                        + str(t.score)
                        + "}"
                        for t in prior
                    ]
                )
                if prior
                else "",
                "allowed_names": "None\n"
                + "\n".join([data_source.name for data_source in data_sources]),
            }
        )
        if reply.data_source == "None":
            return None
        else:
            data_source = next(x for x in data_sources if x.name == reply.data_source)
            print(
                "SELECTION: " + reply.data_source,
                "\nSELECTION_REASON: ",
                reply.reason,
                "\nsteps: ",
                reply.what_is_required_to_answer_the_question,
            )
            return DataSourceSelection(
                data_source=data_source,
                reason=reply.reason,
            )
