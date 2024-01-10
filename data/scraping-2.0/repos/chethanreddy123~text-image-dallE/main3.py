import os
import openai
import streamlit as st

st.title("Python Code Explainer")

text = st.text_area("Copy your python code here: ")



openai.api_key = os.getenv("API_KEY")

check = st.button("Click Here")


if check == True:
    response = openai.Completion.create(
    model="code-davinci-002",
    prompt=text,
    temperature=0,
    max_tokens=64,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )

    st.text_area("The python code explanation is" , response.choices[0].text)

    print(response.choices[0]['text'])
