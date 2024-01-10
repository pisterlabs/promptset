from config import get_OpenAI
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

import streamlit as st

# Set the API key for OpenAI
try:
    OpenAI.api_key = get_OpenAI()
except Exception as e:
    raise Exception(f"Error setting API key for OpenAI: {e}")


# LLMs
llm_model = "gpt-3.5-turbo-1106"
chat = ChatOpenAI(temperature=0.6, model=llm_model)
open_ai = OpenAI(temperature = 0.0)

# Chains
prompt = PromptTemplate(
    input_variables=["language"],
    template="How do you say good afternoon in {language}?",
)

chain = LLMChain(llm=open_ai, prompt=prompt)

st.title("Chain:")
st.text(chain.prompt.template.format(language="Portugese"))
st.write(chain.run(language="Portugese"))
