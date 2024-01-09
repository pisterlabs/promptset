import os 
from apikeys import OPENAI_API_KEY, SERPER_API_KEY, APIFY_API_TOKEN

import streamlit as st 
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper 

from langchain.agents import load_tools
from langchain.agents import initialize_agent

from langchain.document_loaders.base import Document
from langchain.indexes import VectorstoreIndexCreator
from langchain.utilities import ApifyWrapper


os.environ['OPENAI_API_KEY'] = OPENAI_API_KEY
os.environ['SERPAPI_API_KEY'] = SERPER_API_KEY
os.environ["APIFY_API_TOKEN"] = APIFY_API_TOKEN


st.title('Let\'s find out More...')
st.divider()
prompt = st.text_input('Please type in your question(s)')


llm = OpenAI(temperature=0.9)
tools = load_tools(["wikipedia", "serpapi"], llm=llm)

agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)

if prompt:
    response = agent.run(prompt)
    st.write(response)




