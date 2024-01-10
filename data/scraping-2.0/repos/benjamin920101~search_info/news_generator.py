# Importing necessary packages, files and services
import os

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

os.environ['OPENAI_API_KEY'] = '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'

# App UI framework
st.title('🦜🔗 Tweet Generator')
prompt = st.text_input('Tweet topic: ')

# Prompt templates
title_template = PromptTemplate(
    input_variables = ['topic'],
    template='給我寫一則關於 {topic} 的推文'
)

tweet_template = PromptTemplate(
    input_variables = ['title', 'wikipedia_research'],
    template='寫一篇關於此標題的推文標題給我： {title} ，同時利用此維基百科：{wikipedia_research} '
)

# Wikipedia data
wiki = WikipediaAPIWrapper()

# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
tweet_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# Llms
llm = OpenAI(model_name="text-davinci-003", temperature=0.9)
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
tweet_chain = LLMChain(llm=llm, prompt=tweet_template, verbose=True, output_key='script', memory=tweet_memory)

# Chaining the components and displaying outputs
if prompt:
    title = title_chain.run(prompt)
    wiki_research = wiki.run(prompt)
    tweet = tweet_chain.run(title=title, wikipedia_research=wiki_research)

    st.write(title)
    st.write(tweet)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Tweet History'):
        st.info(tweet_memory.buffer)

    with st.expander('Wikipedia Research'):
        st.info(wiki_research)