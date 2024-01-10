import streamlit as st
import openai
from openai import OpenAI
from functions import create_storydetails as sdetails

client = OpenAI(api_key = st.secrets.openai.api_key)


def app_wizard_AI():
    st.write("AI Story Builder")
    btnGetStoryDetails = st.button("Get Story Details", key="btnStoryDetails")
    if btnGetStoryDetails:
        if "storydatadetails" not in st.session_state:
            storydetails = sdetails.create_story_details(st.session_state.storydata)
            st.session_state.storydatadetails = storydetails
            st.write(st.session_state.storydatadetails)
        else:
            st.write(st.session_state.storydatadetails)
    
   

