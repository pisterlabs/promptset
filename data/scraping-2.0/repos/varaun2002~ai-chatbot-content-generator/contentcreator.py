
#installing libraries
import openai
import gradio
import os
import streamlit as st
import re
import pandas as pd

#open ai api key
#for github
openai.api_key=st.secrets["OPENAI_API_KEY"]
#for local
# openai.api_key=os.getenv("OPENAI_API_KEY")

messages=[{"role":"user","content":"you are content creator"}]
def customchatgpt(user_input):
    messages.append({"role":"user","content":user_input})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages)
    Chatgptreply=response["choices"][0]["message"]["content"]
    messages.append({"role":"assistant","content":Chatgptreply})
    return Chatgptreply


def main():
    st.title("AI Chatbot for Content Creation")
    
    # Text input
    input_text = st.text_input("Enter text")
    
    # Submit button
    if st.button("Submit"):
        # Call the process_text function
        result = customchatgpt(input_text)


        # Display the result
        st.write("result: ", result)


if __name__ == "__main__":
    main()
