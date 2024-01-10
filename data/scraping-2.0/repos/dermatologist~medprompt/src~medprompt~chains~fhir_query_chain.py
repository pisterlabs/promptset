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


from langchain.output_parsers import GuardrailsOutputParser
from langchain.prompts.prompt import PromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.tools import tool
from overrides import override
from .base_medprompt_chain import BaseMedpromptChain


class FhirQueryChain(BaseMedpromptChain):
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
        self._name = "question to fhir query"
        self.init_prompt()


    @override
    def init_prompt(self):
        if "rail_spec" in self._prompt:
            rail_spec = self._prompt["rail_spec"]
        else:
            self.med_prompter.set_template(template_name='fhir_rails_v1.xml')
            rail_spec = self.med_prompter.generate_prompt()
        output_parser = GuardrailsOutputParser.from_rail_string(rail_spec, api=self.main_llm)
        self.FHIR_QUERY_PROMPT = PromptTemplate(
            template=output_parser.guard.prompt.escape(),
            input_variables=["question"],
        )


    @property
    @override
    def chain(self):
        """Get the runnable chain."""
        _cot = RunnablePassthrough.assign(
            question = lambda x: x["question"],
            ) | self.FHIR_QUERY_PROMPT | self.main_llm | StrOutputParser()
        chain = _cot.with_types(input_type=self.input_type)
        return chain


@tool(FhirQueryChain().name, args_schema=FhirQueryChain().input_type)
def get_tool(**kwargs):
    """
    Returns the FHIR query string.
    The input is a dict with the following keys:
        question (str): The question asked to the model.
        resource (str): The FHIR resource to query.
    """
    return FhirQueryChain().chain.invoke(kwargs)