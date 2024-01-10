import streamlit as st
import openai
from dotenv import load_dotenv
import os
import pyperclip

# Function to copy text to clipboard
def copy_to_clipboard(text):
    pyperclip.copy(text)
    st.success("Text copied to clipboard!")

# Set your OpenAI API key
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to get completion from OpenAI
def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,  # Adjust this for randomness
    )
    return response.choices[0].message["content"]

# Streamlit app
st.title("Mail Maker")

# Create selection boxes for 'To:' and 'word_limit'
recipient = st.selectbox("To:", ['HOD', 'FRIEND', 'CLASS TEACHER', 'HEAD'])
word_limit = st.selectbox("Word Limit:", ['30-50', '50-70', '70-100', '100-120'])

# Text input box
text = st.text_area("Enter your text here:")

s = ''  # Initialize s outside of the if block

# Button to generate email
if st.button("Generate Email"):
    # Prepare the prompt based on user's choices
    prompt = f"""
    TASKS:
    Your task is to generate the text based on the given text and compress it to {word_limit} words, and that email is to {recipient}.

    Do the above tasks for text, delimited by triple backticks

    Review: ```{text}```
    """

    # Get completion
    response = get_completion(prompt)
    s = response  # Assign the generated email to s

    # Display the generated email
    st.subheader("Generated Email:")
    st.text_area("Email", response, height=300)

    if st.button("Copy to Clipboard"):
        copy_to_clipboard(response)

# import pyperclip
# text_to_be_copied = 'The text to be copied to the clipboard.'
# pyperclip.copy(text_to_be_copied)