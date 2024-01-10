import os
import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

# V3 - using mmemory with Sequential chains

from apikey import OPENAI_API_KEY
os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY

# App Framework
st.title('🦜️🔗 Youtube Script generator (GPT)')
prompt = st.text_input('Type in your prompt')

# prompt template 
title_template = PromptTemplate(input_variables = ['topic'], 
                                template = 'write me a youtube video title about {topic}')
script_template = PromptTemplate(input_variables = ['title', 'wiki_research'], 
                                 template = 'write me a youtube video script based on this title. TITLE: {title} while leveraging this wikipedia research: {wiki_research}')

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# LLMs 
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose = True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose = True, output_key='script', memory=script_memory)
wiki = WikipediaAPIWrapper()

# Display response if there is a prompt 
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wiki_research = wiki_research)

    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'):
        st.info(wiki_research)

# # V2 - using multiple outputs in app
# # 
# # # App Framework
# st.title('🦜️🔗 Youtube Script generator (GPT)')
# prompt = st.text_input('Type in your prompt')

# # prompt template 
# title_template = PromptTemplate(input_variables = ['topic'], 
#                                 template = 'write me a youtube video title about {topic}')
# script_template = PromptTemplate(input_variables = ['title', 'wiki_research'], 
#                                  template = 'write me a youtube video script based on this title. TITLE: {title} while leveraging this wikipedia research: {wiki_research}')

# # LLMs 
# llm = OpenAI(temperature=0.9)
# title_chain = LLMChain(llm=llm, prompt=title_template, verbose = True, output_key='title', memory=title_memory)
# script_chain = LLMChain(llm=llm, prompt=script_template, verbose = True, output_key='script', memory=script_memory)
# # order matters in chains 
# sequential_chain = SequentialChain(
#     chains=[title_chain, script_chain],
#     input_variables=['topic'],
#     output_variables=['title', 'script'],
#     verbose = True)
# wiki = WikipediaAPIWrapper()

# # Display response if there is a prompt 
# if prompt:
#     response = sequential_chain({'topic': prompt})
#     st.write(response['title'])
#     st.write(response['script'])
# # =====

# # V1
# only outputs the last output of the sequential chain 

# # SimpleSequentialChain 
# # App Framework
# st.title(' Youtube GPT Creator')
# prompt = st.text_input('Type in your prompt')

# # prompt template 
# title_template = PromptTemplate(
#     input_variables = ['topic'],
#     template = 'write me a youtube video title about {topic}'
# )

# script_template = PromptTemplate(
#     input_variables = ['title'],
#     template = 'write me a youtube video script based on this title. TITLE: {title}'
# )

# # LLMs 
# llm = OpenAI(temperature=0.9)
# title_chain = LLMChain(llm=llm, prompt=title_template, verbose = True)
# script_chain = LLMChain(llm=llm, prompt=script_template, verbose = True)
# sequential_chain = SimpleSequentialChain(chains=[title_chain, script_chain], verbose = True)


# # Display response if there is a prompt 
# if prompt:
#     # response = llm(prompt)
#     # response = title_chain.run(topic=prompt)
#     response = sequential_chain.run(prompt)
#     st.write(response)
