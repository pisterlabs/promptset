import streamlit as st
from openai import OpenAI
from util_toast import get_toast_message

def generate_story_details(story_data):
    """
    Generates the storybook title and summary using GPT-3.5turbo16k.
    :param character_description: Description of the main character
    :param story_elements: Dictionary of story elements like theme, tone, etc.
    :return: Tuple of story title and summary
    """
    tst_start = get_toast_message("start", "Story Details")

    client = OpenAI(api_key=st.secrets.openai.api_key)
    assistant_id = st.secrets.openai.assistant_key_magicbook

    #create thread
    thread_id = client.beta.threads.create().id
    st.session_state.thread_id = thread_id

    instructions = f"""
    Create a 10-page storybook outline, narrative, and illustration prompts using the following details:

    {story_data}

    REMINDER: Only return the final json output.
    """

    