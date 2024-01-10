import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st

os.environ['OPENAI_API_KEY'] = openai_key

st.title('Celebrity Search Engine')
input_text = st.text_area('Search topic u want to know about')

llm = OpenAI(temperature=0.8)

# Memory
person_memory = ConversationBufferMemory(
    input_key='name', memory_key='chat_history')
dob_memory = ConversationBufferMemory(
    input_key='dob', memory_key='chat_history')
events_memory = ConversationBufferMemory(
    input_key='events', memory_key='description_history')

# PromptTemplate
first_prompt = PromptTemplate(
    input_variables=['name'],
    prompt="Tell me about celebrity {name}",
)
chain1 = LLMChain(llm=llm, prompt=first_prompt,
                  verbose=True, output_key='person', memory=person_memory)

second_prompt = PromptTemplate(
    input_variables=['person'],
    prompt="when was {person} born?",
)
chain2 = LLMChain(llm=llm, prompt=second_prompt,
                  verbose=True, output_key='dob', memory=dob_memory)

third_prompt = PromptTemplate(
    input_variables=['dob'],
    prompt="Mention 5 major events that happened in {dob} in the world",
)
chain3 = LLMChain(llm=llm, prompt=third_prompt,
                  verbose=True, output_key='events', memory=events_memory)

parent_chain = SequentialChain([chain1, chain2],
                               input_variables=['name'],
                               output_variables=['person', 'dob', 'events'],
                               verbose=True)


if input_text:
    st.write(parent_chain.run(input_text))

    with st.expander('Person Memory'):
        st.info(person_memory.buffer)

    with st.expander('DOB Memory'):
        st.info(dob_memory.buffer)

    with st.expander('Events Memory'):
        st.info(events_memory.buffer)
