# Purpose:chatbot for excel data
#author: Likun Yang
#date: 2023-5-30

import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI

@st.cache_data
def load_data(file):
    if file.name.endswith(".xlsx"):
        df = pd.read_excel(file)
    elif file.name.endswith(".csv"):
        df = pd.read_csv(file)
    return df

@st.cache_data
def load_model():
    OPENAI_API_KEY = "your key"
    llm = OpenAI(api_token=OPENAI_API_KEY)
    pandas_ai = PandasAI(llm)
    return pandas_ai

pandas_ai = load_model()

col1, col2 = st.columns(2)

st.sidebar.title("DataGPT")
#add upload file in sidebar
uploaded_file = st.sidebar.file_uploader("上传Excel文件", type=["csv", "xlsx"])


if uploaded_file is not None:
    df = load_data(uploaded_file)
    # Initialization  ################
    if 'df' not in st.session_state:
        st.session_state['df'] = df
    # if 'df_tmp' not in st.session_state:
    #     st.session_state['df_tmp'] = df
    ##############################  
    with col1: 
        st.header("原数据")
        st.dataframe(st.session_state.df)
    
    prompt = st.text_input("Prompt")

    if st.button("SUBMIT"):
        answer = pandas_ai.run(st.session_state.df, prompt=prompt)
        st.markdown(f"**Answer:** {answer}")
        #diff = st.session_state.df.compare(st.session_state.df_tmp)
        #st.session_state.df_tmp = st.session_state.df
        with col2:
            st.header("新数据")
            st.dataframe(st.session_state.df)
