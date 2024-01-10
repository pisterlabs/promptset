import streamlit as st
# from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper


st.title(':speech_balloon: Chat')

llm = OpenAI(temperature=0.7, max_tokens=200, openai_api_key=st.secrets["openai"]["api_key"])

num_purposes = st.text_input("Number of purposes", key="purpose")

if not num_purposes:
    num_purposes = 1

# Prompt template
motivation_template = PromptTemplate(
    input_variables = ['num_purposes', 'context', 'wikipedia_search'],
    # template = "Write me a motivational message about ```{context}``` based on while leveraging this wikipedia search: {wikipedia_search}"
    template = "Suggest {num_purposes} me a motivational message about ```{context}``` based on while leveraging this wikipedia search: {wikipedia_search}"
)

wiki = WikipediaAPIWrapper()
# motivation_theory = "growth mindset"
motivation_theory = "stress-is-enhancing mindset"
wikipedia_search = wiki.run(motivation_theory)

motivation_memory = ConversationBufferMemory(input_key="context", memory_key="chat_history")

motivation_chain = LLMChain(llm=llm, prompt=motivation_template, verbose=True, output_key="motivation_msg", memory=motivation_memory)


def generate_response(prompt):
  if prompt:
    res = motivation_chain.run(num_purposes=num_purposes, context=prompt, wikipedia_search=wikipedia_search)
    st.write(res)

with st.form('my_form'):
  text = st.text_area('Enter text:', '')
  submitted = st.form_submit_button('Submit')
  if submitted:
    generate_response(text)
#   if not openai_api_key.startswith('sk-'):
#     st.warning('Please enter your OpenAI API key!', icon='⚠')
#   if submitted and openai_api_key.startswith('sk-'):
#     generate_response(text)