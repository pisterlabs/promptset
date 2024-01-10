"""Handler functions for the components"""
import streamlit as st
import openai
import gnosis.gui_messages as gm


def set_api_key():
    """Set the OpenAI API key."""
    openai.api_key = st.session_state.api_key
    st.session_state.api_message = gm.api_message(openai.api_key)


def click_wk_button():
    """Set the OpenAI API key."""
    st.session_state.wk_button = not st.session_state.wk_button
