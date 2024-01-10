'''
If you want your searches to be of category type . It should be specific and not generic.
You will have to use Prompt Engineering for Custom Use cases 

Combining multiple Prompts 
i/p - Tell me about Virat Kohli - o/p
Above o/p should be given to 2nd prompt template

'''


## Integrate our Code with Open AI API
import os 
from constants import openai_key
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
## Whenever you use prompt template . YOu will have to use below 
from langchain.chains import LLMChain ## It is responsible for executing this prompt template 

## Combining Multiple chains 
from langchain.chains import SimpleSequentialChain
## To show information of all chains 
from langchain.chains import SequentialChain

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
## output_key used to have mupltiple chaning of prompt templates 
chain = LLMChain(llm=llm,prompt=first_input_prompt,verbose=True,output_key='person')

## Second Prompt Template 

second_prompt_template = PromptTemplate(
    input_variables=['person'],
    template='when was {person} born'
)

chain2 = LLMChain(llm=llm,prompt=second_prompt_template,verbose=True,output_key='dob')

## Third Prompt Template 

third_prompt_template = PromptTemplate(
    input_variables=['dob'],
    template='Mention 5 major events happended around {dob} in the world'
)

chain3 = LLMChain(llm=llm,prompt=third_prompt_template,verbose=True,output_key='description')

## Problem of SimpleSeuentialChain is that it shows last output of chain .
#parent_chain = SimpleSequentialChain(chains=[chain,chain2],verbose=True)
parent_chain = SequentialChain(chains=[chain,chain2,chain3],
                               input_variables=['name'],
                               output_variables=['person','dob','description'],
                               verbose=True)

if input_text:
    #st.write(parent_chain.run(input_text)) ## For SimpleSequentialChain 
    st.write(parent_chain({
        'name' : input_text
    }))
## In order to run streamlit applications 
## > streamlit run E1_CombineMultiplePrompts.py 
## So whatever you write in text box will be added with prompt and then it is sent to api
