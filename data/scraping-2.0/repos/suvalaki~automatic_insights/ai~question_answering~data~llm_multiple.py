from typing import List, Tuple, Optional
from abc import ABC, abstractmethod
from collections import OrderedDict

from pydantic import BaseModel, Field
from langchain import PromptTemplate, LLMChain
from langchain.tools import Tool
from langchain.base_language import BaseLanguageModel
from langchain.output_parsers import PydanticOutputParser

from ai.question_answering.schema import (
    Thought,
    Hypothesis,
    DataSourceSelection,
    MultipleDataSourceSelection,
)
from ai.question_answering.data.base import DataSourceSelector
from ai.question_answering.data.llm_single import DatasourceSelectorQuery


class MultipleDatasourceSelectorQuery(BaseModel):
    # Plan is a Chain Of Thought
    plan: List[str] = Field(
        description="A list of steps to used to solve the hypothesis using the data.."
    )
    plan_objective: str = Field(
        description="The purpose of the selections made in the plan."
    )
    selection: Tuple[DatasourceSelectorQuery, ...] = Field(
        [],
        description="The data sources used to answer the plan. "
        "Each plan step usually correspond to a new selection.",
    )


PROMPT = """
You are tasked with evaluating a set of data_sources (and data getting functions) which could be used to answer a hypothesis.
It is your goal to select which sources to use, and to explain why you selected them. 
Your selection can be part of a multiple step strategy where you combine multiple data sources to calculate an answer to the hypothesis.
You may also evaluate if the hypothesis is not answerable with any data_source (or combinations of them) provided. 


The hypothesis is: 
{hypothesis}


The data_sources are:
{data_sources}



Select the whole name of the data_source.
If the hypothesis is not answerable with any data_source, you should reply with an empty list as the data_source instead.



You should take account of the data you already have so that you dont select the same data. 
Be exploratory in your selection.
It is not enough to be relevant to the hypothesis. The selection must actually be useful in answering the hypothesis too.
Factor in data which is already selected in the past based on the thoughts. You dont need data you already have.
Select only the needed data. You want to select as few data sources as required to explore the hypothesis.



Prior explored data_source discussions:
{prior}
Only select data sources that provides new information.


{format_instructions}


The set of allowable names are: 
{allowed_names}

Answer only with one selected data_source. The name must be exactly as is provided in this input.
You must only reply with the name of one data_source or "None".

"""


OUTPUT_PARSER = PydanticOutputParser(pydantic_object=MultipleDatasourceSelectorQuery)


PROMPT = PromptTemplate(
    template=PROMPT,
    input_variables=["hypothesis", "data_sources", "prior", "allowed_names"],
    partial_variables={"format_instructions": OUTPUT_PARSER.get_format_instructions()},
    output_parser=OUTPUT_PARSER,
)


class LLMMultipleDataSourceSelector(DataSourceSelector):
    def __init__(self, data_sources: List[Tool], llm: BaseLanguageModel, **kwargs):
        super().__init__(data_sources)
        self.chain = LLMChain(
            llm=llm,
            prompt=PROMPT,
            output_parser=OUTPUT_PARSER,
        )

    def _parse_single(
        self, reply: DatasourceSelectorQuery, data_sources
    ) -> DataSourceSelection:
        if not reply:
            return None
        elif reply.data_source == "None":
            return None

        data_source = next(x for x in data_sources if x.name == reply.data_source)
        return DataSourceSelection(
            data_source=data_source,
            reason=reply.reason,
        )

    def _parse_multiple(
        self, reply: MultipleDatasourceSelectorQuery, data_sources
    ) -> MultipleDataSourceSelection:
        ordered = OrderedDict()
        print(reply)
        for s in reply.selection:
            if s.data_source in ordered.keys():
                ordered[s.data_source].reason += (
                    "\n" + self._parse_single(s, data_sources).reason
                )
            else:
                ordered[s.data_source] = self._parse_single(s, data_sources)
        selections = list(ordered.values())
        return MultipleDataSourceSelection(
            objective=reply.plan_objective + "\n\n" + str(reply.plan),
            selection=selections,
        )

    def __call__(
        self, hypothesis: Hypothesis, prior: List[Thought]
    ) -> MultipleDataSourceSelection | None:
        data_sources = hypothesis.data_sources
        reply = self.chain.predict(
            **{
                "hypothesis": hypothesis.hypothesis,
                "data_sources": "\n".join(
                    [str(data_source) for data_source in data_sources]
                ),
                "prior": "\n".join([str(t) for t in prior]) if prior else "",
                "allowed_names": "None\n"
                + "\n".join([data_source.name for data_source in data_sources]),
            }
        )

        return self._parse_multiple(reply, hypothesis.data_sources)
