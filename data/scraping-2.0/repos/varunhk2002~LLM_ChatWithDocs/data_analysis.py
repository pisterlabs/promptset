import streamlit as st
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.agents import create_pandas_dataframe_agent
import pandas as pd
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv, find_dotenv





if __name__ == "__main__":
    load_dotenv(find_dotenv(), override=True)
    file_name = None

    st.image('img.jpg')
    st.subheader('Analyse your CSV files!!')

    with st.sidebar:
        uploaded_file = st.file_uploader('Upload a file:', type=['csv'])
        add_data = st.button('Add Data')

        if uploaded_file and add_data:
            with st.spinner('Uploading File'):
                bytes_data = uploaded_file.read()
                file_name = os.path.join('./', uploaded_file.name)
                with open(file_name, 'wb') as f:
                    f.write(bytes_data)
                st.session_state.fl = file_name
            
    # print(file_name)
    # doc = pd.read_csv(f'{file_name}', index_col=0)
    # print(doc)
    # chat = ChatOpenAI(model_name='gpt-4', temperature=0.0)
    # agent = create_pandas_dataframe_agent(chat, doc, verbose=True)
    # st.session_state.ag = agent

    q = st.text_input('Ask a question about the content of your file: ')

    if q:
        if 'fl' in st.session_state:
            file_name = st.session_state.fl
            doc = pd.read_csv(f'{file_name}', index_col=0)
            chat = ChatOpenAI(model_name='gpt-4', temperature=0.0)
            agent = create_pandas_dataframe_agent(chat, doc, verbose=True)
            answer = agent.run(q)
            st.text_area('LLM Answer: ', value=answer)

