from langchain.chains.llm import LLMChain
from langchain.llms.base import BaseLLM
from langchain.output_parsers.regex import RegexParser

from langchain.prompts import PromptTemplate


output_parser = RegexParser(
    regex=r"Question\s?\d?:\s+\n?(.*?)\nA(.*?)\nCHOICE_B(.*?)\nCHOICE_C(.*?)\nCHOICE_D(.*?)(?:\n)+Answer:\s?(.*)\n?\n?Question\s?\d?:\s+\n?(.*?)\nCHOICE_A(.*?)\nCHOICE_B(.*?)\nCHOICE_C(.*?)\nCHOICE_D(.*?)(?:\n)+Answer:\s?(.*)",
    output_keys=["question1", "A_1", "B_1", "C_1", "D_1", "reponse1",
                 "question2", "A_2", "B_2", "C_2", "D_2", "reponse2"]
)

PROMPT = PromptTemplate(
    input_variables=["doc"], template=template
)


class QuizChain(LLMChain):

    @classmethod
    def from_llm(cls, llm, **kwargs):
        """Load QA Generate Chain from LLM."""
        return cls(llm=llm, prompt=PROMPT, **kwargs)
