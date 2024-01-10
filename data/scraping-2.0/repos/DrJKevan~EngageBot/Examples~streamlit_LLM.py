# This is an example of integrating a LLM with streamlit
import streamlit as st
import os
import openai
import langchain
from langchain.llms import OpenAI
from langchain import PromptTemplate
#from dotenv import load_dotenv

# Specify the path to the .env file
#dotenv_path = os.path.join(os.path.dirname(__file__), '.env')

# Load the .env file
#load_dotenv(dotenv_path)

# Streamlit Code
st.set_page_config(page_title="Globalize Email", page_icon = ":robot:")
st.header("Globalize Text")

# LLM Code
template = """
    Below is an email that is poorly worded.
    Your goal is to:
    - Properly format the email
    - Convert the input text to a specified tone
    - Convert the input text to a specified dialect
    
    Here are some examples of different Tones:
    - Formal: We went to Barcelona for the weekend. We have a lot of things to tell you.
    - Information: Went to Barcelona for the weekend. Lots to tell you.
    
    Here are some examples of words in different dialects:
    - American English: French Fries, cotton candy, apartment, garbage cookie
    - British English: chips, candyfloss, flag, rubbish, biscuit
    
    Below is the email, tone, and dialect:
    TONE: {tone}
    DIALECT: {dialect}
    EMAIL: {email}
    
    YOUR RESPONSE:
"""
prompt = PromptTemplate(
    input_variables=["tone", "dialect", "email"],
    template = template,
)

llm = OpenAI(temperature = .5)

col1, col2 = st. columns(2)

with col1:
    st.markdown("This application is a demo of the SRL chatbot being developed between UCATT and UAHS International")

with col2:
    st.image(image='screenshot.png', width=500, caption="Screenshot of source video")

st.markdown("## Enter Your Email to Convert")

col1, col2 = st.columns(2)
with col1:
    option_tone = st.selectbox(
        'Which tone would you like your email to have?',
        ('Formal', 'Informal'))
    
with col2:
    option_dialect = st.selectbox(
        'Which English Dialect would you like?',
        ('American English', 'British English')
    )

def get_text():
    input_text = st.text_area(label="", placeholder = "Your email...", key="email_input")
    return input_text

email_input = get_text()

st.markdown("### Your Converted Email:")

if email_input:
    prompt_with_email = prompt.format(tone = option_tone, dialect = option_dialect, email = email_input)
    
    # See full prompt
    #st.write(prompt_with_email)

    formatted_email = llm(prompt_with_email)

    st.write(formatted_email)
