"""Sidebar component for the Streamlit app."""
import streamlit as st
import openai
from gnosis.components.handlers import set_api_key, click_wk_button


def delete_collection(client, collection):
    """Delete collection button."""
    if st.button("Delete collection"):
        st.warning("Are you sure?")
        if st.button("Yes"):
            try:
                client.delete_collection(collection.name)
            except AttributeError:
                st.error("Collection erased.")


def openai_api_key_box():
    """Box for entrying OpenAi API Key"""
    st.sidebar.write("## OpenAI API key")
    openai.api_key = st.sidebar.text_input(
        "Enter OpenAI API key",
        value="",
        type="password",
        key="api_key",
        placeholder="Enter your OpenAI API key",
        on_change=set_api_key,
        label_visibility="collapsed",
    )
    st.sidebar.write(
        "You can find your API key at https://platform.openai.com/account/api-keys"
    )


def creativity_slider():
    """Slider with temperature level"""
    st.sidebar.subheader("Creativity")
    st.sidebar.write("The higher the value, the crazier the text.")
    st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.25,  # Max level is 2, but it's too stochastic
        value=0.5,
        step=0.01,
        key="temperature",
    )


def wk_checkbox():
    """Wikipedia Checkbox for changing state"""
    st.sidebar.checkbox(
        "Use Wikipedia", on_change=click_wk_button, value=st.session_state.wk_button
    )


# Sidebar
def sidebar(client, collection):
    """Sidebar component for the Streamlit app."""
    with st.sidebar:
        openai_api_key_box()

        wk_checkbox()

        creativity_slider()

        delete_collection(client, collection)
