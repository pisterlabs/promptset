# Bring in deps
import os 
import sys
sys.path.append('../')

from LLM_Playground import apikeys

import streamlit as st 
from langchain import HuggingFaceHub
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

os.environ['OPENAI_API_KEY'] = apikeys.openai_key
os.environ['HUGGINGFACEHUB_API_TOKEN'] = apikeys.huggingface_key

# App framework
st.title('🦜🔗 Explain Like a fifth grader?')
prompt = st.text_input('Plug in your prompt here') 

script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'], 
    template='Explain as if you were a fifth grader based on this title TITLE: {title} while leveraging this wikipedia reserch:{wikipedia_research}'
)

# Memory
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# Llms
#llm = OpenAI(model='gpt-3.5-turbo', temperature=0.9) 
llm = llm=HuggingFaceHub(repo_id="tiiuae/falcon-7b-instruct", model_kwargs={"temperature":0.9})
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper(top_k_results=1)

# Show stuff to the screen if there's a prompt
if prompt: 
    wiki_research = wiki.run(prompt) 
    script = script_chain.run(title=prompt, wikipedia_research=wiki_research)

    st.write(script) 

    with st.expander('Script History'): 
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'): 
        st.info(wiki_research)