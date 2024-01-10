"""
TestGenius - Code Testing and Test Case Generation

Empowers developers to create reliable and comprehensive test suites effortlessly. TestGenius uses AI
to generate test cases for code snippets, functions, or classes, fostering correctness and enhancing
test coverage. This accelerates the development cycle while ensuring robust software quality.
"""
from typing import Type
import streamlit as st
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import (
    ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate)
from data.testing_libraries import TESTING_LIBRARIES
from prompts.generate_tests_prompt import create_test_generation_prompt

def show_test_page(chat: Type[ChatOpenAI]):
    """
    Display the TestGenius page with a title, description, and code and test case input form.

    Parameters:
    - openai_api_key (str): API key for accessing the OpenAI GPT-3.5 model.
    """

    st.title("TestGenius - Code Testing and Test Case Generation")

    st.markdown('Empowers developers to create reliable and comprehensive test suites effortlessly. '
                'TestGenius uses AI to generate test cases for code snippets, functions, or classes, '
                'fostering correctness and enhancing test coverage. This accelerates the development '
                'cycle while ensuring robust software quality.')

    with st.form(key="test_form"):
        # Provide examples as placeholders in text areas
        code_snippet = st.text_area(
            "Enter code snippet (e.g., def add_numbers(a, b): return a + b)", value="", height=200)

        selected_testing_library = st.selectbox(
            "Select Testing Library", TESTING_LIBRARIES)

        submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            st.text(f"Creating Tests... ✨")

            chat_prompt = create_test_generation_prompt(selected_testing_library,code_snippet)

            chain = LLMChain(llm=chat, prompt=chat_prompt)
            result = chain.run(code_snippet=code_snippet)

            st.markdown(result)
