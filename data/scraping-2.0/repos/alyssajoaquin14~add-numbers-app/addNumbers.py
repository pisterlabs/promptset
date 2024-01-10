import streamlit as st
import os
import openai

openai.api_key = os.getenv['OPENAI_API_KEY']

st.title("Add Some Numbers")

col1, col2 = st.columns(2)

with col1:
    number1 = st.number_input("Number 1", value = None, placeholder = "Type a number..")

with col2:
    number2 = st.number_input("Number 2", value = None, placeholder = "Type a number..")

if st.button("Add the two numbers together"):
    number3 = number1 + number2
    st.write("Sum = ", number3)
    

if st.button("Ask a LLM for answer"):
    response = openai.chat.completions.create(
        model = "gpt-3.5-turbo",
        messages = [
            {"role": "system", "content": "You are a helpful assistant designed to add numbers together."},
            {"role": "user", "content": f"What is the sum of {number1} and {number2}?"}
        ]
    )
    llmResponse = response['choices'][0]['message']['content']
    st.write(f"The sum is {llmResponse}")