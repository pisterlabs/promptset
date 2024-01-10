from enum import Enum
from typing import List

from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser, RetryWithErrorOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field


class LLM(Enum):
    OPENAI = "OPENAI"
    LLAMA = "LLAMA"


class Concept(BaseModel):
    concepts: List[str] = Field(description="generated_concepts")


class ConceptGenerator:
    def __init__(self, number_of_new_concepts):
        self.generators = {}
        self.number_of_new_concepts = number_of_new_concepts
        self.parser = PydanticOutputParser(pydantic_object=Concept)
        self.base_prompt = PromptTemplate(
            template='''
You are a vocabulary expert that suggests new words to students based on their requests.
Suggest {number_of_new_concepts} unique words to study.
The words should be extracted and transformed in their lexeme from the following text delimited by triple quotes if
it is provided """{text}""" and should match the following description delimited by triple quotes if it is provided
"""{description}""".
The words should be different from the words in this list delimited by triple quotes: """{exclude_concepts}""".
The words should be in the {language} language even if words previously suggested are in other languages.
{format_instructions}''',
            input_variables=[
                "language",
                "number_of_new_concepts",
                "description",
                "text",
                "exclude_concepts",
            ],
            partial_variables={
                "format_instructions": self.parser.get_format_instructions()
            },
        )

    def initialize_generator(self, llm, temperature=0.7):
        if llm == LLM.OPENAI:
            return ChatOpenAI(
                # model_name="gpt-3.5-turbo",
                model_name="gpt-4",
                temperature=temperature,
            )

    def generate_concepts(self, llm, language, description, text, exclude_concepts):
        prompt_kwargs = dict(
            language=language,
            number_of_new_concepts=self.number_of_new_concepts,
            description=description,
            text=text,
            exclude_concepts=exclude_concepts,
        )
        if llm not in self.generators:
            self.generators[llm] = self.initialize_generator(llm)
        llm_chain = LLMChain(prompt=self.base_prompt, llm=self.generators[llm])
        fix_parser = RetryWithErrorOutputParser.from_llm(
            llm=self.generators[llm], parser=self.parser, max_retries=2
        )
        answer = llm_chain.run(**prompt_kwargs)
        prompt_value = self.base_prompt.format_prompt(**prompt_kwargs)
        answer = fix_parser.parse_with_prompt(answer, prompt_value)
        return answer.concepts
