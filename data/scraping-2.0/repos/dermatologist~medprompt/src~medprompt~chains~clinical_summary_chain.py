"""
 Copyright 2023 Bell Eapen

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""


import json
import logging
from typing import List
from langchain.output_parsers import GuardrailsOutputParser
from langchain.prompts.prompt import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableMap, RunnablePassthrough
from langchain.tools import tool
from langchain_core.pydantic_v1 import BaseModel, Field
from overrides import override
from ..tools.expand_concepts import ExpandConceptsTool
from .base_medprompt_chain import BaseMedpromptChain
_logger = logging.getLogger(__name__)


class ClinicalSummaryChain(BaseMedpromptChain):

    class ClinicalConceptInput(BaseModel):
        clinical_document: str = Field()
        word_count: str = Field()

    def __init__(self,
                 chain=None,
                 prompt={},
                 main_llm=None,
                 clinical_llm=None,
                 sec_llm=None,
                 input_type=None,
                 output_type=None
                 ):
        super().__init__(
                chain = chain,
                prompt = prompt,
                main_llm = main_llm,
                clinical_llm = clinical_llm,
                sec_llm = sec_llm,
                input_type = input_type,
                output_type = output_type,
                )
        self._input_type = self.ClinicalConceptInput
        self._name = "clinical summary generator"
        self.init_prompt()

    @override
    def init_prompt(self):
        rail_spec = """
            <rail version="0.1">
            <output>
                <object name="response" format="length: 1">
                    <string
                        name="concepts"
                        description="The list of concepts extracted from the clinical document."
                        format="list"
                        on-fail-length="reask"
                    />
                </object>
            </output>


            <prompt>

            Given the following clinical document, extract the single main clinical concept and return it as a single item list.
            If the concept has two words, return them as a single word joined by an underscore.

            Clinical document: ${clinical_document}
            ${gr.complete_json_suffix_v2}

            </prompt>
            </rail>
        """

        if "rail_spec" in self._prompt:
            rail_spec = self._prompt["rail_spec"]

        clinical_concepts_output_parser = GuardrailsOutputParser.from_rail_string(rail_spec, api=self.main_llm)

        self.CLINICAL_CONCEPT_INPUT_PROMPT = PromptTemplate(
            template=clinical_concepts_output_parser.guard.prompt.escape(),
            input_variables=["clinical_document"],
        )

        CLINICAL_CONCEPT_SUMMARY_TEMPLATE = """
        You are summarizing the clinical document below. These {clinical_concepts} may be included in the summary.

        Clinical Document: {clinical_document}.
        Summary:"""

        if "CLINICAL_CONCEPT_SUMMARY_TEMPLATE" in self._prompt:
            CLINICAL_CONCEPT_SUMMARY_TEMPLATE = self._prompt["CLINICAL_CONCEPT_SUMMARY_TEMPLATE"]

        self.CLINICAL_CONCEPT_SUMMARY_PROMPT = PromptTemplate(
            template=CLINICAL_CONCEPT_SUMMARY_TEMPLATE,
            input_variables=["clinical_concepts", "clinical_document"],
        )

    def extract_concepts(self, guardrails_output):
        """Extract the concepts from the clinical document."""
        _gr = json.loads(guardrails_output)
        return _gr["response"]

    def concat_concepts(self, concepts: List[str]):
        """Concatenate the concepts."""
        return " ".join(concepts)

    @property
    @override
    def chain(self):
        """Get the runnable chain."""
        list_of_concepts = RunnablePassthrough.assign(
            clinical_document=lambda x: x["clinical_document"],
        ) | self.CLINICAL_CONCEPT_INPUT_PROMPT | self.main_llm | StrOutputParser() | self.extract_concepts |  ExpandConceptsTool().run | self.concat_concepts
        clinical_document = RunnablePassthrough.assign(
            clinical_document = lambda x: x["clinical_document"] + "\n" + "Summarize to " + x["word_count"] + " words. Summary: ",
        )
        _inputs = RunnableMap(
            clinical_concepts=list_of_concepts,
            clinical_document=clinical_document,
        )
        _chain = _inputs | self.CLINICAL_CONCEPT_SUMMARY_PROMPT | self.main_llm | StrOutputParser()
        chain = _chain.with_types(input_type=self.input_type)
        return chain

@tool(ClinicalSummaryChain().name, args_schema=ClinicalSummaryChain().input_type)
def get_tool(**kwargs):
    """
    Summarize the clinical document to a given word count.
    The input is a dict with the following keys:
        clinical_document (str): The clinical document to summarize.
        word_count (str): The number of words to summarize to.
    """
    return ClinicalSummaryChain().chain.invoke(kwargs)