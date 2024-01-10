import streamlit as st
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
import os
from dotenv import load_dotenv

#  getting the enviroment variables
# load_dotenv()
# OPEN_AI_API_KEY=os.getenv('OPEN_AI_API_KEY')


# function to generate response
def generate_response(text):
    # model initialse
    llm=OpenAI(temperature=0,openai_api_key=OPEN_AI_API_KEY)
    #  splitting of text 
    text_splitter = CharacterTextSplitter()
    texts = text_splitter.split_text(text)

    docs = [Document(page_content=t) for t in texts]
    # Text summarization
    chain = load_summarize_chain(llm, chain_type='map_reduce')
    return chain.run(docs)

# setting up the page 
st.set_page_config(page_title='🦜🔗 Summary Generator')
st.title('🦜🔗 Summary Generator')

# Text input
text_input = st.text_area('Enter the text to be summarized', '', height=200)


result = []
with st.form('summarize_form', clear_on_submit=True):
    OPEN_AI_API_KEY = st.text_input('OpenAI API Key', type = 'password', disabled=not text_input)
    submitted = st.form_submit_button('Submit')
    if submitted and OPEN_AI_API_KEY.startswith('sk-'):
        with st.spinner('Calculating...'):
            response = generate_response(text_input)
            result.append(response)
            del OPEN_AI_API_KEY


if len(result):
    st.info(response)

