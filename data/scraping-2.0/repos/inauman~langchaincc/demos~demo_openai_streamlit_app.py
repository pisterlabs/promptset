import os

import streamlit as st
from langchain.llms import OpenAI
from langchain.chains import LLMChain
from langchain import PromptTemplate


st.set_page_config(page_title="🦜🔗 Blog Outline Generator App")
st.title('🦜🔗 Blog Outline Generator App')

def generate_response(topic):
    llm = OpenAI(model_name='text-davinci-003')
    # Prompt
    template = 'As an experienced data scientist and technical writer, generate an outline for a blog about {topic}.'
    prompt = PromptTemplate(input_variables=['topic'], template=template)
    # prompt_query = prompt.format(topic=topic)
    # Run LLM model and print out response
    # response = llm(prompt_query)

    # setup and run the chain
    outline_chain = LLMChain(llm=llm, prompt=prompt,
                             output_key='outline', verbose=True)
    response = outline_chain.run(topic=topic)
    return st.info(response)


with st.form('myform'):
    topic_text = st.text_input('Enter keyword:', '')
    submitted = st.form_submit_button('Submit')

    if submitted:
        generate_response(topic_text)
