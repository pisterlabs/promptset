import streamlit as st

from langchain.utilities import WikipediaAPIWrapper
from langchain.tools import DuckDuckGoSearchRun

from langchain.agents import load_tools, initialize_agent, AgentType
from langchain.llms import OpenAI 

import os
import sys
sys.path.append('_private/')
from _private import api
os.environ['OPENAI_API_KEY'] = api.API_KEY

# Duckduckgo search
search = DuckDuckGoSearchRun()

# Import all PDFs in folder
import glob
import re

# Get all PDFs in folder
path = 'pdfs/'
pdfs = glob.glob(path + '*.pdf')
#df = pd.concat([pd.read_csv(file) for file in file_paths], ignore_index=True)


# Init the LLM and Tools
llm = OpenAI(temperature=0)
tools = load_tools(['wikipedia'], llm=llm)
agent = initialize_agent(
    tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)


# collect user prompt
st.title('SullyGPT')
input = st.text_input('Enter a prompt:')

if input:
#    text = search.run(input)
    text = agent.run(input)
    st.write(text)
