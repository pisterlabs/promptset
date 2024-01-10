import os
import streamlit as st
from pandasai import SmartDataframe
from pandasai.callbacks import BaseCallback
from pandasai.llm import OpenAI
from pandasai.responses.response_parser import ResponseParser
import pandas as pd

openai_api_key = os.getenv("openai_api_key") 

class StreamlitCallback(BaseCallback):
    def __init__(self, container) -> None:
        """Initialize callback handler."""
        self.container = container

    def on_code(self, response: str):
        self.container.code(response)

class StreamlitResponse(ResponseParser):
    def __init__(self, context) -> None:
        super().__init__(context)

    def format_dataframe(self, result):
        st.dataframe(result["value"])
        return

    def format_plot(self, result):
        st.image(result["value"])
        return

    def format_other(self, result):
        st.write(result["value"])
        return


st.write("# Chat with Credit Card Fraud Dataset 🦙")

uploaded_file = st.file_uploader("Upload a Dataset", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    with st.expander("🔎 Dataframe Preview"):
        st.write(df.head())

query = st.text_area("🗣️ Chat with Dataframe")
container = st.container()

if query:
    llm = OpenAI(api_token=openai_api_key)
    query_engine = SmartDataframe(
        df,
        config={
            "llm": llm,
            "response_parser": StreamlitResponse,
            "callback": StreamlitCallback(container),
        },
    )

    answer = query_engine.chat(query)
    st.write(answer)
