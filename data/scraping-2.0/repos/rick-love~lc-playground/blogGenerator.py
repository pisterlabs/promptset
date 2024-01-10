from OpenAI_Training.config import get_OpenAI
from openai import OpenAI
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper
from langchain.chat_models import ChatOpenAI
from langchain.globals import get_verbose
current_verbose = get_verbose()

import streamlit as st

# Set the API key for OpenAI
try:
    OpenAI.api_key = get_OpenAI()
except Exception as e:
    raise Exception(f"Error setting API key for OpenAI: {e}")

# LLMs
llm = OpenAI(temperature=0.9)
chat_model = ChatOpenAI()

#Memory
blog_title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
blog_subject_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')

# Prompt Templates
prompt = st.text_input('Enter your topic:')
blog_title_template = PromptTemplate(
    input_variables=['topic'],
    template='Write a Blog title about {topic}'
)
blog_subject_templte = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template='Write a Blog article based on this title: {title} while also leveraging this wikipedia research: {wikipedia_research}. The article should be less 500 words long.'
)

title_chain= LLMChain(llm=llm, prompt=blog_title_template, output_key='title', memory=blog_title_memory)
subject_chain= LLMChain(llm=llm, prompt=blog_subject_templte, output_key='title', memory=blog_subject_memory)

wiki = WikipediaAPIWrapper()


############
# Show to the screen
# App Framework
st.title('Blog Creator')
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    subject = subject_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title)
    st.write(subject)

    with st.expander('Title History'):
        st.info(blog_title_memory.buffer)
    with st.expander('Script History'):
        st.info(blog_subject_memory.buffer)
    with st.expander('Wikipedia Research'):
        st.info(wiki_research)


