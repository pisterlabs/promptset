"""
RefactorRite - Code Refactoring Advisor

Leverage AI-driven code analysis and automated refactoring to enhance code
readability, boost performance, and improve maintainability. RefactorRite
suggests intelligent refinements and even automates the refactoring process,
allowing developers to focus on building robust software.
"""
from typing import Type
import streamlit as st
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from prompts.refactor_code_prompt import create_refactoring_prompt

def show_refactor_page(chat: Type[ChatOpenAI]):
    """
    Display the RefactorRite page with a title, description, and code input form.

    Parameters:
    - openai_api_key (str): API key for accessing the OpenAI GPT-3.5 model.
    """

    st.title("RefactorRite - Code Refactoring Advisor")

    st.markdown(
        """
        Leverage AI-driven code analysis and automated refactoring to enhance code
        readability, boost performance, and improve maintainability. RefactorRite
        suggests intelligent refinements and even automates the refactoring process,
        allowing developers to focus on building robust software.
        """
    )

    with st.form(key="refactor_form"):
        # Allow users to enter a code snippet in a text area
        code_snippet = st.text_area("Enter code snippet")

        # Create a submit button within the form
        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            st.text(f"Refactoring code snippet... ✨")

            chat_prompt = create_refactoring_prompt(code_snippet)
            # Initialize an LLMChain for running the AI conversation
            chain = LLMChain(llm=chat, prompt=chat_prompt)

            # Get the result of the refactoring suggestions from the AI
            result = chain.run(code_snippet=code_snippet)

            # Display the result of the refactoring suggestions
            st.text_area("Refactor suggestions", result, height=400)
