# Import Libraries
import os
from dotenv import load_dotenv
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
import pandas as pd
from pandasai import PandasAI
from pandasai import SmartDataframe
from pandasai.middlewares import StreamlitMiddleware
from pandasai.llm import OpenAI
import matplotlib.pyplot as plt

# API Key retrieval
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="Data Analysis Tool", page_icon="🔧")

# Sidebar contents
with st.sidebar:
    st.title('Pandas-AI 🐼')
    st.markdown('''
    ## About
    This app is a data analysis tool created using:
    - ### [Streamlit](https://streamlit.io/)
    - ### [Pandas](https://pandas.pydata.org/docs/)
    - ### [PandasAI](https://github.com/gventuri/pandas-ai)
    - ### [OpenAI LLM model](https://platform.openai.com/docs/models)

    ''')
    add_vertical_space(20)
    st.write('Made with ❤️ by [Jayesh_Ironside](https://github.com/jayeshironside)')

# Front-end and Back-end starts here
st.title("Prompt-driven analysis tool 🔧")

uploaded_file = st.file_uploader("Upload a CSV file for analysis", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file, encoding='Latin-1')
    df = SmartDataframe(data, {"middlewares": [StreamlitMiddleware()]})
    st.write(data.head(3))
    st.write("Rows x Column :", df.shape)

    prompt = st.chat_input("Enter your prompt here...")
    if prompt:
        with st.spinner("Generating you response..."):
            llm = OpenAI(api_token=OPENAI_API_KEY)
            pandas_ai = PandasAI(llm, verbose=True)
            x = pandas_ai.run(df, prompt=prompt)
            fig = plt.gcf()
            if fig.get_axes():
                st.pyplot(fig)
            st.write("👉 Generated Response :", x)