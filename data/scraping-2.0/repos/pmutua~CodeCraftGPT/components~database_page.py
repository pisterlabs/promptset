from typing import Type
import streamlit as st
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from data.database_System import Database_system
from prompts.translate_code_prompt import create_translation_prompt

def show_database_page(chat: Type[ChatOpenAI]):
    """
    Displays and generates the Database query.

    Parameters:
    - chat (Type[ChatOpenAI]): The ChatOpenAI instance.

    Returns:
    None
    """
    st.title("Database - Database Query Question")

    st.markdown('Database AI Tool ')

    # Append a unique identifier to the form key to make it unique
    unique_key = "lang_form_" + 'database'
    
    with st.form(key=unique_key):  # Use the unique key for the form
        source_code = st.text_area("Enter Your Question to Generate the Query")
        target_language = st.selectbox("Select Database You Want", Database_system)

        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            st.text(f"Translating code snippet to {target_language}................✨")

            chat_prompt = create_translation_prompt(target_language, source_code)

            chain = LLMChain(llm=chat, prompt=chat_prompt)
            result = chain.run(source_code=source_code, target_language=target_language)
            
            st.text_area("Translated Code", result, height=400)
