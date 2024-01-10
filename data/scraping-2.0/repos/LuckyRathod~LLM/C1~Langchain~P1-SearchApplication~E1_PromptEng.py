'''
If you want your searches to be of category type . It should be specific and not generic.
You will have to use Prompt Engineering for Custom Use cases 
'''


## Integrate our Code with Open AI API
import os 
from constants import openai_key
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
## Whenever you use prompt template . YOu will have to use below 
from langchain.chains import LLMChain ## It is responsible for executing this prompt template 

import streamlit as st
os.environ["OPENAI_API_KEY"] = openai_key

## Initialize Streamlit framework 
st.title('Celebrity Search Results with OpenAI API')
input_text = st.text_input('Search the topic you want')

## Prompt Templates 
first_input_prompt = PromptTemplate(
    input_variables = ['name'],  ## celebrity name 
    template = "Tell me about {name}" ## So you have added prompt template for whatever name you enter 
)

## OpenAI LLMS
## temperature - How much control agent should have while providing the response.
## How much balanced ans you want is determined by temperature
llm = OpenAI(temperature=0.8)
## w.r.t every prompt template you will have LLM Chain 
chain = LLMChain(llm=llm,prompt=first_input_prompt,verbose=True)

if input_text:
    st.write(chain.run(input_text))

## In order to run streamlit applications 
## > streamlit run E1_PromptEng.py 
## So whatever you write in text box will be added with prompt and then it is sent to api
