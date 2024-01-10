import streamlit as st
from functions import login as lg, pagesetup as ps, salesforce as sf
from streamlit_modal import Modal
import streamlit.components.v1 as components
import openai
from openai import OpenAI
import time
import uuid



#0. Page Config
st.set_page_config("AlmyAI", initial_sidebar_state="collapsed", layout="wide")

#1. Login and Page Setup
if lg.check_authentication():
    ps.set_title("AlmyAI", "Clinical")
    ps.set_page_overview("Clinical Assistant", "**Clinical Assistant** is an AI-based assistant trained to interact with the Alma Clinical team.")

    container0 = st.container()
    with container0:
        email = st.text_input("Enter your Salesforce email", key="tiEmail")
        if email:
            sUserId = sf.get_sfUserID(email)
            st.write(sUserId)
    st.divider()
    container1=st.container()
    with container1:
        tab1, tab2, tab3 = st.tabs(["Trainings", "Opportunities", "Activities"])
        with tab1:
            st.markdown("**My Trainings**")
            dfTrainings = sf.get_trainings(email)
            container1a = st.container()
            with container1a:
                st.dataframe(dfTrainings)
        with tab2: 
            st.markdown("**My Opportunities**")
        with tab3: 
            st.markdown("**My Activities**")
                    

            
