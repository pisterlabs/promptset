from constants import *
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from dotenv import load_dotenv
load_dotenv()


def get_chain() -> LLMChain:
    llm = OpenAI(streaming=True)
    chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            template=prompt_template,
            input_variables=["question", "context"]
        )
    )
    return chain