# Bring in deps
import os 
from apikey import apikey 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('🤖 YouTube Shorts-Script Generator 🤖')
prompt = st.text_input('Plug in your prompt here') 

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'], 
    template='write me a youtube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research', 'context'], 
    template='Given the title "{title}", previous script context "{context}", and this research: {wikipedia_research}, write the next 15-second YouTube video script segment.'
)

# Memory 
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# Llms
llm = OpenAI(temperature=0.9) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper()

# Function to extract context from a script segment
def extract_context(segment_script):
    # Implement your logic here to extract key points or summary
    # For simplicity, this could be the last few lines or key points mentioned in the segment
    return context


# Show stuff to the screen if there's a prompt
if prompt: 
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt) 

    full_script = ""
    context = ""  # Initialize context
    for segment_number in range(1, 5):  # Loop for 4 segments
        segment_script = script_chain.run(title=title, wikipedia_research=wiki_research, context=context)
        full_script += segment_script + "\n"  # Append each segment
        context = extract_context(segment_script)  # Update context for the next segment

    st.write(title) 
    st.write(full_script)  # Display the full script

    st.write(title) 
    st.write(full_script)  # Display the full script

    with st.expander('Title History'): 
        st.info(title_memory.buffer)

    with st.expander('Script History'): 
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)