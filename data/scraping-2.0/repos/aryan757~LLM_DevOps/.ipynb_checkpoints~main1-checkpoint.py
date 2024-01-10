import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

from langchain.memory import ConversationBufferMemory

from langchain.chains import SequentialChain

import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key
from langchain import PromptTemplate
demo_template='''I want you to act as a acting financial advisor for people.
In an easy way, explain the basics of {financial_concept}.'''

prompt = PromptTemplate(
    input_variables = ['financial_concept'],
    template = demo_template
)

prompt.format(financial_concept = 'income tax')


from langchain.llms import OpenAI

from langchain.chains import LLMChain

llm = OpenAI()

chain1 = LLMChain(llm = llm , prompt = prompt)

chain1.run('income tax')




