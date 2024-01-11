import streamlit as st

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

# App framework
st.title("YouTube Title Creator")
prompt = st.text_input("Enter your prompt here!")

# Prompt Templates
title_template = PromptTemplate(
    input_variables=["topic"], template="write me a youtube video title about {topic}"
)

script_template = PromptTemplate(
    input_variables=["title"],
    template="write me a youtube video description based on this title TITLE:{title}",
)


# LLMs
llm = OpenAI(temperature=0.9)

# chains
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key="title")
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key="script")

sequential_chain = SequentialChain(
    chains=[title_chain, script_chain],
    input_variables=["topic"],
    output_variables=["title", "script"],
    verbose=True,
)

if prompt:
    response = sequential_chain({"topic": prompt})
    st.write(response["title"])
    st.write(response["script"])