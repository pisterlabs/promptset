#bring in deps
import os
from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ["OPENAI_API_KEY"] = apikey

# App framework
st.title("🦜🔗Youtube GPT creator")
prompt = st.text_input("Plug in your prompt here")

# prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'],
    template="Write me a youtube video title about: {topic}"
)
script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'],
    template="Write me a youtube video script based on this title TITLE: {title} while leaveraging this wikipedia research: {wikipedia_research}"
)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# llms
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)
# sequential_chain = SequentialChain(chains=[title_chain, script_chain], input_variables=['topic'], output_variables=['title', 'script'], verbose=True)

wiki = WikipediaAPIWrapper()

# show stuff if theres a prompt
if prompt:
    title = title_chain.run(prompt)
    wiki_research=wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)
    # response = sequential_chain({'topic': prompt})
    st.write(title)
    st.write(script)

    with st.expander("title history"):
        st.info(title_memory.buffer)
    with st.expander("script history"):
        st.info(script_memory.buffer)
    with st.expander("Wikipedia Research"):
        st.info(wiki_research)