# pylint: disable=invalid-name
""" A Streamlit app for GnosisPages. """
import os
import streamlit as st
import openai
from dotenv import load_dotenv
from gnosis.chroma_client import ChromaDB
import gnosis.gui_messages as gm
from gnosis import settings
from gnosis.components.sidebar import sidebar
from gnosis.components.main import main


load_dotenv()


openai.api_key = os.getenv("OPENAI_API_KEY")

if "api_message" not in st.session_state:
    st.session_state.api_message = gm.api_message(openai.api_key)


if "wk_button" not in st.session_state:
    st.session_state.wk_button = False


# Build settings
chroma_db = ChromaDB(openai.api_key)
collection = settings.build(chroma_db)

# Sidebar
sidebar(chroma_db, collection)

main(openai.api_key, chroma_db, collection)
