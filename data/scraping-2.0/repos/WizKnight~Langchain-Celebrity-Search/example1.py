## Integrating coe with openai API
import os
from constants import openai_key
from langchain.chains import LLMChain
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain

from langchain.chains import SequentialChain

import streamlit as st

os.environ["OPENAI_API_KEY"] = openai_key

# Streamlit framework
st.title('Celebrity Search Results')
input_text = st.text_input("Search the topic")

# Prompt Templates

first_input_prompt = PromptTemplate(
        input_variables = ['name'],
        template = "Tell me about {name}"
        )

## OPENAI LLM Models
# This shows how much control an agent should have while providing you the response.
llm = OpenAI(temperature=0.8)

chain = LLMChain(llm=llm, prompt=first_input_prompt, verbose=True, output_key='name')

# Prompt Templates

second_input_prompt = PromptTemplate(
        input_variables = ['person'],
        template = "Birth Date of {person}"
        )

chain2 = LLMChain(llm=llm, prompt=second_input_prompt, verbose=True, output_key='dob')
parent_chain = SequentialChain(chains=[chain,chain2],
                                input_variables=['name'],output_variables=['person','dob'],verbose=True)



if input_text:
        st.write(parent_chain.run({'name': input_text}))