import streamlit as st

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

# App framework
st.title("YouTube Title Creator")
prompt = st.text_input("Enter your prompt here!")

# Prompt Templates
title_template = PromptTemplate(
    input_variables=["topic"], template="write me a youtube video title about {topic}"
)

script_template = PromptTemplate(
    input_variables=["title", "wikipedia_research"],
    template="write me a youtube video description based on this title TITLE:{title}\
        whilst leveraging this wikipedia research : {wikipedia_research}",
)

# Memory
title_memory = ConversationBufferMemory(input_key="topic", memory_key="chat_history")
script_memory = ConversationBufferMemory(input_key="title", memory_key="chat_history")

# LLMs
llm = OpenAI(temperature=0.9)

# chains
title_chain = LLMChain(
    llm=llm, prompt=title_template, verbose=True, output_key="title", memory=title_memory
)

script_chain = LLMChain(
    llm=llm, prompt=script_template, verbose=True, output_key="script", memory=script_memory
)

wiki = WikipediaAPIWrapper()


if prompt:
    title = title_chain.run(topic=prompt)
    wiki_research = wiki.run(prompt)
    script = script_chain.run(title=title, wikipedia_research=wiki_research)
    st.write(title)
    st.write(script)

    with st.expander("Title History"):
        st.info(title_memory.buffer)

    with st.expander("Script History"):
        st.info(script_memory.buffer)

    with st.expander("Wikipedia Reserach"):
        st.info(wiki_research)
