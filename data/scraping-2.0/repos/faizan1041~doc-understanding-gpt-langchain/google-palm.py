from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens
from langchain.llms import GooglePalm
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
from config.config import *
import os

from langchain import PromptTemplate, LLMChain

import google.generativeai as genai

genai.configure(api_key=os.environ['GOOGLE_API_KEY'])


fact_llm = GooglePalm(temperature=0.1)

from langchain import PromptTemplate, HuggingFaceHub, LLMChain

template = """Question: {question}

Answer: Let's think step by step."""

prompt_open = PromptTemplate(template=template, input_variables=["question"])

open_chain = LLMChain(prompt=prompt_open,llm = fact_llm)

question = "What NFL team won the Super Bowl in the year Justin Beiber was born?"

print(open_chain.run(question))