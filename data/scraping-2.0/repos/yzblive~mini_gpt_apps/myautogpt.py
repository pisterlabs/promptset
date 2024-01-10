# ref: https://www.youtube.com/watch?v=MlK6SIjcjE8&list=LL&index=4
from dotenv import load_dotenv
import os
import streamlit as st

from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.utilities import WikipediaAPIWrapper

def main():
    os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
    # load_dotenv()
    # print(os.getenv('OPENAI_API_KEY'))
    st.set_page_config(page_title="My AutoGPT")
    # st.header('My AutoGPT')

    st.title('Youtube GPT Creator')
    prompt = st.text_input('Plug in your prompt here')

    title_template = PromptTemplate(
        input_variables = ['topic'],
        # template='write me a youtube video title about {topic}'
        template='给我写一个bilibili视频标题，关于{topic}，用中文写'
    )

    script_template = PromptTemplate(
        input_variables = ['title', 'wikipedia_research'],
        # template='write me a youtube video script based on this title: {title} while leveraging this wikipedia research: {wikipedia_research}'
        template='基于这个标题"{title}"给我写一个bilibili视频脚本，并参考以下来自于wikipedia的调查信息: {wikipedia_research}，用中文写，不少于2000字'
    )


    title_memory = ConversationBufferMemory(input_key='topic',memory_key='chat_history')
    script_memory = ConversationBufferMemory(input_key='title',memory_key='chat_history')

    llm = OpenAI(temperature=0.9)
    title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
    script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)
    # seq_chain = SequentialChain(chains=[title_chain, script_chain], 
    #                                   input_variables=['topic'], 
    #                                   output_variables=['title', 'script'], verbose=True)

    wiki = WikipediaAPIWrapper()

    if prompt:
        # resp = llm(prompt)
        # resp = title_chain.run(topic=prompt)
        # resp = seq_chain.run(prompt)
        # st.write(resp)

        # resp = seq_chain({'topic': prompt})
        with get_openai_callback() as cb:
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

            print(cb)

if __name__ == '__main__':
    main()