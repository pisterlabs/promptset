# #use dotenv to get the api keys from .env file
# from dotenv import load_dotenv,find_dotenv
# load_dotenv(find_dotenv())

import os
from langchain import PromptTemplate, OpenAI, LLMChain
import chainlit as cl

os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
template = """Question: {question}

Answer: Let's think step by step."""

@cl.langchain_factory(use_async=True)
def factory():
    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0), verbose=True)

    return llm_chain


#enter "chainlit run app.py -w" in terminal to run application. 
#the -w is to enable auto-reloading so you don't need to restart server any time you make changes
