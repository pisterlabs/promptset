import os 
from apikey import apikey 

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

os.environ['OPENAI_API_KEY'] = apikey

# App framework
st.title('🦜🔗 App code GPT Creator')
prompt = st.text_input('Plug in your prompt here') 

# Prompt templates
code_template = PromptTemplate(
    input_variables = ['topic'], 
    template="""You're an expert python programmer AI Agent. You generate Python code based on {topic}.
You should ALWAYS output the full detailed code."""
)

# Memory 
code_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')

# Llms
llm = OpenAI(temperature=0.9) 
code_chain = LLMChain(llm=llm, prompt=code_template, verbose=True, output_key='title', memory=code_memory)


# Show stuff to the screen if there's a prompt
if prompt: 
    code = code_chain.run(prompt)
    source_code_start = code.find('Source Code:\n') + len('Source Code:\n')
    source_code_end = code.find('Thought:')
    source_code = code[source_code_start:source_code_end].strip()
    st.write('Extracted Source Code:')
    st.code(source_code)
    st.write(code)

    with st.expander('Code History'): 
        st.info(code_memory.buffer)
    
    with open('output.py', 'w') as file:
        file.write(code)
