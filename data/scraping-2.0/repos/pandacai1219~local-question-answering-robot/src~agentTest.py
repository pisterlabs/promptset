
from keys import OpenAI_API_KEY
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from template import prompt_template_Document, prompt_template_GPT
import streamlit as st
import os
from langchain.chains import LLMChain,SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ["OPENAI_API_KEY"] = OpenAI_API_KEY
st.title("🦜️🔗Youtube GPT")
prompt = st.text_input("Enter your prompt here")

#Prompt templates
title_template = PromptTemplate(
    input_variables=['topic'],
    template="write me a youtube video title about {topic}"
)
#Prompt templates
script_template = PromptTemplate(
    input_variables=['title','wikipedia_research'],
    template="write me a youtube video script based on this title TITLE: {title} while leveraging the following research: {wikipedia_research}"
)
#Memory
title_memory = ConversationBufferMemory(input_key='topic',memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title',memory_key='chat_history')
    # Initialize the LLM
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, verbose=True, prompt=title_template,output_key='title',memory=title_memory)
script_chain = LLMChain(llm=llm, verbose=True, prompt=script_template,output_key='script',memory=script_memory)
#sequential_chain = SequentialChain(chains=[title_chain,script_chain],input_variables=['topic'],output_variables=['title','script'],verbose=True)

wiki = WikipediaAPIWrapper()

if prompt:
    #response = sequential_chain({'topic':prompt})
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title,wikipedia_research=wiki_research)
    st.write(title)
    st.write(script)
    
    with st.expander('Ttile History'):
        st.info(title_memory.buffer)
    with st.expander('Script History'):
        st.info(script_memory.buffer)
    
    with st.expander('Wikipedia Research'):
        st.info(wiki_research)