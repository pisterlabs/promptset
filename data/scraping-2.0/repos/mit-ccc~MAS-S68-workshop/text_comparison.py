#!/usr/bin/env python3

"""
Code for the MAS.S68 (Generative AI for Constructive Communication) programming workshop

Simple text similarity tool, using the OpenAI embeddings API
"""

import os

import numpy as np
import openai
import streamlit as st

# Don't forget to set your OPENAI_API_KEY environment variable.
# Or set it here directly (but don't check it into a git repo.)
openai.api_key = os.getenv("OPENAI_API_KEY")


@st.cache_data
def call_embeddings_api(prompt, engine="text-embedding-ada-002"):
    response = openai.Embedding.create(engine=engine, input=prompt)
    embedding = response["data"][0]["embedding"]
    return embedding


def streamlit_app():
    with st.form("main_form"):
        text1 = st.text_input("Enter text 1")
        text2 = st.text_input("Enter text 2")
        submitted = st.form_submit_button("Submit")
        if submitted:
            embedding1 = call_embeddings_api(text1)
            embedding2 = call_embeddings_api(text2)
            with st.expander("Embedding 1"):
                st.write(embedding1)
            with st.expander("Embedding 2"):
                st.write(embedding2)
            cosine_sim = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            st.write("Cosine similarity is: %0.4f" % (cosine_sim))


if __name__ == "__main__":
    streamlit_app()
