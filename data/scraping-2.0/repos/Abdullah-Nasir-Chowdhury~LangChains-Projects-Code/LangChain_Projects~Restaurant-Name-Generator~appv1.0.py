import os, openai
from dotenv import main

main.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
print(openai.api_key)

import streamlit as st
st.title('Restaurant Name Generator')
st.write("Generate restaurant names on the fly!")


from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
llm = OpenAI(temperature=0)
name_template = PromptTemplate(
    input_variables = ['cuisine'],
    template = '''Your task is to generate a name for the restaurant that serves {cuisine} food'''
)
name_chain = LLMChain(llm=llm, 
                      prompt=name_template,
                      output_key = 'name')
menu_template = PromptTemplate(
    input_variables = ['name'],
    template = '''Your task is to generate a menu for a restaurant whose name is {name}.
    You will make sure that the menu is related to the name of the restaurant. 
    Your answer will be in numbers.
    Add 4 sections in your answer with each section of the menu having at least 3 items. 
    The sections are: Appetizers, Main Course, Desserts and Drinks.'''
)
menu_chain = LLMChain(llm=llm, 
                      prompt=menu_template,
                      output_key = 'menu')
sequential_chain = SequentialChain(chains=[name_chain, menu_chain], 
                                   verbose=True,
                                   input_variables=['cuisine'],
                                   output_variables=['name','menu'])

# ouputs:
prompt = st.text_input("Plug in your cuisine", placeholder='e.g. Indian')
if prompt:
    response = sequential_chain({'cuisine':f'{prompt}'})
    print(response)
    st.write(response)
