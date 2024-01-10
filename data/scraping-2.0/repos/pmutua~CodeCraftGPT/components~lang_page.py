"""
LangLink - Code Translation and Cross-Language Compatibility

Overcome language barriers with LangLink, an AI-powered tool facilitating smooth code translation
between programming languages. Developers can confidently migrate codebases, ensuring compatibility
and seamless transitions across different languages.
"""
from typing import Type
import streamlit as st
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from data.programming_languages import PROGRAMMING_LANGUAGES
from prompts.translate_code_prompt import create_translation_prompt

def show_lang_page(chat: Type[ChatOpenAI]):
    """
    Displays the LangLink page for code translation.

    Parameters:
    - openai_api_key (str): The API key for OpenAI.

    Returns:
    None
    """
    st.title("LangLink - Code Translation and Cross-Language Compatibility")

    st.markdown('Overcome language barriers with LangLink, an AI-powered tool facilitating smooth '
                'code translation between programming languages. Developers can confidently migrate '
                'codebases, ensuring compatibility and seamless transitions across different languages.')

    with st.form(key="lang_form"):
        source_code = st.text_area("Enter source code")
        target_language = st.selectbox("Select programming language", PROGRAMMING_LANGUAGES)

        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            
            st.text(f"Translating code snippet to {target_language}................✨")

            chat_prompt = create_translation_prompt(target_language,source_code)

            chain = LLMChain(llm=chat, prompt=chat_prompt)
            result = chain.run(source_code=source_code,
                               target_language=target_language)
            
            st.text_area("Translated Code", result, height=400)
            
