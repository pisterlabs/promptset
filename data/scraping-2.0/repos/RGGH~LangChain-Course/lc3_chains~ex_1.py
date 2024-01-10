import streamlit as st

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain

# App framework
st.title("YouTube Video Title Creator")
prompt = st.text_input("Enter your prompt here!")

# Prompt Templates
title_template = PromptTemplate(
    input_variables = ["topic"],
    template="Create a YouTube video title about {topic}"
)

script_template = PromptTemplate(
    input_variables = ["title"],
    template="Write a YouTube video description \
        for me based on this title TITLE:{title}"
)


# # LLM
# llm = OpenAI(temperature=0.9)

# # Chains
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True)

sequential_chain = SimpleSequentialChain(chains=[title_chain,script_chain])

if prompt:
    response = sequential_chain.run(prompt)
    st.write(response)
    
    