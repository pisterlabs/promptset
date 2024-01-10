import streamlit as st
import pandas as pd
import numpy as np
import time
from utils import openai_api

st.set_page_config(page_title="Email Composer", page_icon=None) #, menu_items=None)

st.title("Email Composer")

st.text_area(
    "Email Message",
    key="message",
    placeholder="Past your email here.",
    height=600,
)

st.text_area(
    "Instructions for Generation",
    key="instructions",
    placeholder="E.g. accept, 15 euros per hour, remote from Rotterdam.",
)

model = st.radio(label="Model", options=["gpt-3.5-turbo", "gpt-4"])

generate_response = st.button("Generate Email")

message = st.session_state.message
instructions = st.session_state.instructions

response_col1, response_col2 = st.columns(2)

if generate_response:
    response = openai_api.response_generator(
        emails_to_respond_to=message,
        response_instructions=instructions,
        model=model,
    )

    with response_col1:
        st.text_area(
            label="Response I",
            value=response["Email_1"],
            height=300,
        )

    with response_col2:
        st.text_area(
            label="Response II",
            value=response["Email_2"],
            height=300,
        )

else:
    with response_col1:
        st.text_area(
            label="Response I",
            value="",
            height=300,
        )

    with response_col2:
        st.text_area(
            label="Response II",
            value="",
            height=300,
        )

# hide_streamlit_style = """
#             <style>
#             [data-testid="stToolbar"] {visibility: hidden !important;}
#             footer {visibility: hidden !important;}
#             </style>
#             """
# st.markdown(hide_streamlit_style, unsafe_allow_html=True)