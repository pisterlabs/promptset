"""
CodeDocGenius - Code Documentation Generator

Automatically generates documentation for code snippets in any programming language.
"""
from typing import Type
import streamlit as st
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)
from prompts.code_documentation_prompt import create_documentation_prompt

def show_doc_page(chat: Type[ChatOpenAI]):
    """
    Display the CodeDocGenius page with a title, description, and code input form.

    Parameters:
    - openai_api_key (str): API key for accessing the OpenAI GPT-3.5 model.
    """

    st.title("CodeDocGenius - Code Documentation Generator")
    
    st.markdown('Automatically generates documentation for code snippets in any programming language.')

    with st.form(key="doc_form"):
        code_snippet = st.text_area("Enter code snippet")

        submit_button = st.form_submit_button(label='Generate Documentation')
        
        if submit_button:
            
            st.text(f"Generating documentation... ✨")
            
            chat_prompt = create_documentation_prompt(code_snippet)
            chain = LLMChain(llm=chat, prompt=chat_prompt)
            result = chain.run(code_snippet=code_snippet)

            st.text_area("Generated Documentation", result, height=400)
