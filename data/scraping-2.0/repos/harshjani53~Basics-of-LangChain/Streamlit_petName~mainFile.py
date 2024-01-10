
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from dotenv import load_dotenv

load_dotenv()


def generateName(type, color):
    llm = OpenAI(temperature=0.7)

    template = PromptTemplate(
        input_variables = ['type','color'],
        template = "Can you suggest me 5 cool and creative names for my {color} {type}?."
    )

    chain = LLMChain(llm=llm, prompt=template, output_key="name")

    ans = chain({'type': type, 'color': color})


    return ans
