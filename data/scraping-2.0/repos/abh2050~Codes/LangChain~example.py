##integrate our code with OpenAI API
import os
from constant import openai_key
from langchain.llms import OpenAI
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
import streamlit as st


os.environ["OPENAI_API_KEY"] = openai_key

#streamlit framework
st.title('Famous Person Search Results')
input_text = st.text_input('Search the topic you want:')

#prompt template for input
first_input_prompt = PromptTemplate(
    input_variables=['name'],
    template= 'Tell me about Famous Person {name}.',
)

# Memory
person_memory = ConversationBufferMemory(input_key='name',memory_key='chat_history')
dob_memory = ConversationBufferMemory(input_key='person',memory_key='chat_history')
descr_memory = ConversationBufferMemory(input_key='dob',memory_key='description_history')

#OPENAI LLMS
llm = OpenAI(temperature=0.8)
chain = LLMChain(llm=llm, prompt=first_input_prompt,verbose=True,output_key='person',memory=person_memory)

#prompt template for second input
second_input_prompt = PromptTemplate(
    input_variables=['person'],
    template= 'When was {person} born?'
)
#chain 2 for second input
chain2 = LLMChain(llm=llm, prompt=second_input_prompt,verbose=True,output_key='dob',memory=dob_memory)

#prompt template for 3rd input
third_input_prompt = PromptTemplate(
    input_variables=['dob'],
    template= 'Mention 5 major events happened on {dob} in the world'
)
#chain 3 for second input
chain3 = LLMChain(llm=llm, prompt=third_input_prompt,verbose=True,output_key='description',memory=descr_memory)

#prompt template for sequence input
parent_chain = SequentialChain(chains=[chain,chain2,chain3],input_variables=['name'],output_variables=['person','dob','description'], verbose=True)


if input_text:
    st.write(parent_chain({'name':input_text}))

    with st.expander('Person Name'):
        st.write(person_memory.buffer)

    with st.expander('Major Events'):
        st.write(descr_memory.buffer)
    
    




