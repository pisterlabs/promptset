import streamlit as st
from functions import login as lg, pagesetup as ps, salesforce as sf
from streamlit_modal import Modal
import streamlit.components.v1 as components
import pandas as pd
import openai
from openai import OpenAI


#0. Page Config
st.set_page_config("AlmyAI", initial_sidebar_state="collapsed", layout="wide")
if "ClinFileId" not in st.session_state:
    st.session_state.ClinFileId = ""
    
openai.api_key = st.secrets.OPENAI_API_KEY
client = OpenAI()

#1. Login and Page Setup
if lg.check_authentication():
    ps.set_title("AlmyAI", "Clinical")
    ps.set_page_overview("Overview", "**AlmyAI - Clinical** is the first Alma AI-based app.")

