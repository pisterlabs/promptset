from openai import OpenAI
import streamlit as st 
import pandas as pd
import streamlit_pandas as sp


@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('data/titanic.csv')
    return df

df = load_data()
st.write(df)


    