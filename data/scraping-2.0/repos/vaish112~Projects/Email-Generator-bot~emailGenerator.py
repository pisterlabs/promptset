import os
from constants import openai_key
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import SequentialChain, LLMChain
from langchain.memory import ConversationBufferMemory

import streamlit as st

os.environ["OPENAI_API_KEY"]=openai_key

st.title("Email Generator Application")

# inputs for email generation
to_text=st.text_input("Enter your recipient of email")
subject_text=st.text_input("subject for your email")
additional_text=st.text_input("Any additional content to add in email")
manner_text=st.text_input("Professional/Casual/Formal/Informal way of email")

llm=OpenAI(temperature=0.8)

memory_mailBody = ConversationBufferMemory(input_key='to', memory_key='chat_history')
memory_additionalBody = ConversationBufferMemory(input_key='additional', memory_key='chat_history')

first_input_prompt = PromptTemplate(
    input_variables=['to', 'subject'],
    template="write a short email to {to} about {subject}"
    )

# configure OpenAI to llm
llm=OpenAI(temperature=0.5)

chain = LLMChain(
    llm=llm,prompt=first_input_prompt, verbose=True, output_key='mailBody', memory=memory_mailBody
)

# second chain for additionals in mail

second_input_prompt = PromptTemplate(
    input_variables=['additional', 'manner'],
    template="refactor above mail {additional} in {manner} manner.")

chain2 = LLMChain(
    llm=llm, prompt=second_input_prompt, verbose=True, output_key='finalMail', memory=memory_additionalBody)

# combining chains
parent_chain = SequentialChain(
    chains=[chain,chain2],
    input_variables=['to', 'subject', 'additional', 'manner'],
    output_variables=['mailBody', 'finalMail'],
    verbose=True)

if to_text and subject_text and additional_text and manner_text:
    st.write(parent_chain({'to':to_text, 'subject': subject_text, 'additional':additional_text, 'manner': manner_text}))

    with st.expander('Mail body'): 
        st.info(memory_mailBody.buffer)
# if additional_text and manner_text:
#     st.write(parent_chain({'to':additional_text, 'subject': manner_text}))
    with st.expander('Refactored mail'): 
        st.info(memory_additionalBody.buffer)
