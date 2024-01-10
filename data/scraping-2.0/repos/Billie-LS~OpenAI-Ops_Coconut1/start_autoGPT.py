# !pip install streamlit
# !pip install langchain
# !pip install openai
# !pip install wikipedia
# !pip install yfinance
# !pip install chromadb
# !pip install tiktoken

# """Using LangChain plus Streamlit"""

# import deps
import os
import streamlit as st
from langchain.llms import OpenAI
import sys
sys.path.append('/opt/homebrew/lib/python3.10/site-packages')


# from langchain.loaders import DirectoryLoader


key = os.environ.get('apikey')


# from apikey.env import (apikey)

os.environ['OPENAI_API_KEY'] = apikey

st.title('🤖 🚀 ')
