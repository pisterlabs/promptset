import os
import streamlit as st
import pandas as pd
from pandasai import PandasAI
from pandasai.llm.openai import OpenAI
import matplotlib.pyplot as plt
from dotenv import load_dotenv

load_dotenv()


st.title("IA -ciadosorriso ")

st.write("Primeira interface de inteligência artificial da Cia do Sorriso")
st.write(
    "Importe a arquivo csv"
)

openai_key =  os.environ.get('OPENAI_KEY')
key = openai_key
st.session_state.openai_key = openai_key 

#if "openai_key" not in st.session_state:
#    with st.form("API key"):
#        key = openai_key  #st.text_input("OpenAI Key", value="", type="password")
        #if st.form_submit_button("Submit"):
#st.session_state.openai_key = openapi_key

st.session_state.prompt_history = []
st.session_state.df = None

df = pd.read_parquet("../db/df_data_virada.parquet")
st.session_state.df

"""
if "openai_key" in st.session_state:
    if st.session_state.df is None:
        uploaded_file = st.file_uploader(
            "Escolha um arquivo CSV. Deve estar em formato longo (um objeto por linha).",
            type="csv",
        )
        if uploaded_file is not None:
            #df = pd.read_csv(uploaded_file)
            
            st.session_state.df = df
"""
with st.form("o que vc gostaria de saber?"):
        question = st.text_input("o que vc gostaria de saber?", value="", type="default")
        submitted = st.form_submit_button("Submit")
        if submitted:
            with st.spinner():
                llm = OpenAI(api_token=st.session_state.openai_key)
                pandas_ai = PandasAI(llm)
                x = pandas_ai.run(st.session_state.df, prompt=question)

                fig = plt.gcf()
                if fig.get_axes():
                    st.pyplot(fig)
                st.write(x)
                st.session_state.prompt_history.append(question)

if st.session_state.df is not None:
        st.subheader("tabela atual:")
        st.write(st.session_state.df)

st.subheader("Prompt históricos:")
st.write(st.session_state.prompt_history)


if st.button("Clear"):
    st.session_state.prompt_history = []
    st.session_state.df = None
