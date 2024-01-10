from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
import streamlit as st
from prompts import PROMPT_IMPROVER_PROMPT

# This Python Script is a Streamlit App that allows you to generate a prompt using the Prompt Improver Template
# You can run this file by running the following command in your terminal:
# streamlit run prompt_generator_streamlit.py

# Set up Streamlit Interface

with st.container():
    st.markdown("""
                ## Enter initial prompt here:
                """)
initial_prompt = st.text_area(label="Prompt Input", label_visibility='collapsed', placeholder="Generate a workout schedule", key="prompt_input")

if initial_prompt:
    # Initialize LLM
    openai_api_key = "YOUR_OPENAI_API_KEY"
    llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo-16k", temperature=0.4)

    # Initialize LLMChain
    prompt_improver_chain = LLMChain(llm=llm, prompt=PROMPT_IMPROVER_PROMPT)

    # Run LLMChain
    with st.spinner("Generating..."):
        improved_prompt = prompt_improver_chain.run(initial_prompt)
        st.markdown("""
                    ## Improved Prompt:
                    """)
        st.code(improved_prompt)