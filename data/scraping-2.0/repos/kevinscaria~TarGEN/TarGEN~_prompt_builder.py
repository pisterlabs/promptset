from typing import Dict, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.prompts import HumanMessagePromptTemplate
from langchain.output_parsers import CommaSeparatedListOutputParser, NumberedListOutputParser


class PromptBuilder:
    def __init__(self, ):
        self.messages = [SystemMessage(
            content="You are a creative instruction following assistant and your name is TarGEN."
        )]
        self.csv_output_parser = CommaSeparatedListOutputParser()
        self.numbered_list_parser = NumberedListOutputParser()

    def get_contexts(self,
                     step1_prompt: str,
                     partial_variables: Optional[Dict[str, str]] = None
                     ):
        step1_prompt = HumanMessage(content=step1_prompt,
                                    partial_variables=partial_variables)
        step1_prompt = self.messages + [step1_prompt]
        return step1_prompt

    def get_instance_seed(self,
                          step2_prompt: str,
                          domain: str,
                          n: int,
                          partial_variables: Optional[Dict[str, str]] = None
                          ):
        step2_prompt = HumanMessagePromptTemplate.from_template(template=step2_prompt,
                                                                input_variables=["DOMAIN", "N"],
                                                                partial_variables=partial_variables)
        step2_prompt = step2_prompt.format(DOMAIN=domain, N=n)
        step2_prompt = self.messages + [step2_prompt]
        return step2_prompt

    def get_samples(self,
                    step3_prompt: str,
                    sentence: str,
                    partial_variables: Optional[Dict[str, str]] = None
                    ):
        step3_prompt = HumanMessagePromptTemplate.from_template(template=step3_prompt,
                                                                input_variables=["SENTENCE"],
                                                                partial_variables=partial_variables
                                                                )
        step3_prompt = step3_prompt.format(SENTENCE=sentence)
        step3_prompt = self.messages + [step3_prompt]
        return step3_prompt

    def get_self_correction_prompt(self,
                                   step4_prompt: str,
                                   partial_variables: Optional[Dict[str, str]] = None
                                   ):
        step4_prompt = HumanMessagePromptTemplate.from_template(template=step4_prompt,
                                                                partial_variables=partial_variables
                                                                )
        step4_prompt = self.messages + [step4_prompt]
        return step4_prompt
