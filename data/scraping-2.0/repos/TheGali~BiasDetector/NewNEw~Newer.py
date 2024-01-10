import streamlit as st
import openai
import os
import re


# Initialize OpenAI API
# Note: Use environment variables for API keys for security reasons.
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to generate jokes based on the statement
def generate_joke(statement):
    prompt = f"Write a joke in the style of the late Mitch Hedberg with this topic  : {statement}"
    response = openai.ChatCompletion.create(
    model="gpt-4-0613",
    messages=[
    {
      "role": "user",
      "content": prompt
    }
    ],
    temperature=0.7,
    max_tokens=6850,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0
)
    
    return response['choices'][0]['message']['content']


# Title with large font and style
st.markdown("<h1 style='text-align: center; color: blue;'>🔬 Thoughts Laboratory 🔬</h1>", unsafe_allow_html=True)

# Dynamic Textbox for Surveyor
statement = st.text_input('Enter your problem with society here:', '')

# Generate arguments and chart only when a statement is entered
if statement:
    arguments = generate_joke(statement)  # Assuming you have this function defined
    st.title("Joke:")   
    st.write(arguments)
