
import os


import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from openai import OpenAI

os.environ['OPENAI_API_KEY'] = st.secrets["key1"]
client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],)



# App framework
st.title('🦜🔗 YouTube GPT Creator')
prompt = st.text_input('Plug in your prompt here')

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'],
    template='write me a youtube video title about {topic}'
)

script_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'],
    template='write me a youtube video script based on this title TITLE: {title} while leveraging this wikipedia reserch:{wikipedia_research} '
)

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# Llms
llm = OpenAI(temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

wiki = WikipediaAPIWrapper()

# Show stuff to the screen if there's a prompt
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)

    with st.expander('Wikipedia Research'):
        st.info(wiki_research)

##
##os.environ['OPENAI_API_KEY'] = st.secrets["key1"]
##client = OpenAI()
##
##
### Create LangChain with OpenAI as the language model
##
##lang_chain = LLMChain()
##
### Create a conversation memory
##conversation_memory = ConversationBufferMemory(input_key='user_input', memory_key='chat_history')
##
### Streamlit app
##st.title('LangChain Chatbot with Memory')
##
### User input
##user_input = st.text_input('You:')
##
### Generate response when the user clicks a button
##if st.button('Get Response'):
##    # Add user input to conversation history
##    conversation_history = conversation_memory.read()
##    conversation_history.append(user_input)
##    conversation_memory.write(conversation_history)
##
##    # Generate response using LangChain
##    response = lang_chain.run(user_input, memory={'chat_history': conversation_history})
##
##    # Add AI response to conversation history
##    conversation_history.append(response)
##    conversation_memory.write(conversation_history)
##
##    # Display AI response
##    st.text_area('Chatbot:', response, height=100)
##
### Display conversation history
##st.text_area("Conversation History", "\n".join(conversation_memory.read()), height=200)
##
