import os

from dotenv import load_dotenv
import streamlit as st

import vertexai
from vertexai.language_models import ChatModel
import openai

load_dotenv()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "credentials.json"
openai.api_key = os.getenv("OPENAI_API_KEY")


def palm2_chat(prompt):
    vertexai.init(project="llm-ai", location="us-central1")
    chat_model = ChatModel.from_pretrained("chat-bison")
    chat = chat_model.start_chat()
    if prompt:
        response = chat.send_message(prompt)
        return response.text


def gpt4_chat(prompt):
    res = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
    )
    return res['choices'][0]['message']['content']


def judge(prompt, res_from_palm2, res_from_gpt4):
    judge_prompt = f"""
    Determine whether response A or B is more appropriate for the following instruction and output the result. 
    The results should follow the output format.

    Instruction: {prompt}
    Response A: {res_from_palm2}
    Response B: {res_from_gpt4}
    Output format:
        - 0: A is more appropriate
        - 1: B is more appropriate
        - Output should contain nothing but numbers indicating the result.
    """
    res = openai.ChatCompletion.create(
        model="gpt-4-0613",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": judge_prompt},
        ]
    )
    return res['choices'][0]['message']['content']


st.title("PaLM2 vs GPT-4")
st.write("Please enter a prompt and see which LLM is better at responding to it.")

st.header("Prompt")
prompt = st.text_input("Please enter a prompt.")

if st.button("Submit"):
    with st.spinner("PaLM2 is thinking..."):
        res_from_palm2 = palm2_chat(prompt)
        st.header("Response from PaLM2")
        st.write(res_from_palm2)

    with st.spinner("GPT-4 is thinking..."):
        res_from_gpt4 = gpt4_chat(prompt)
        st.header("Response from GPT-4")
        st.write(res_from_gpt4)

    with st.spinner("Judging.."):
        st.header("Judge")
        judge_result = judge(prompt, res_from_palm2, res_from_gpt4)
        if int(judge_result) == 0:
            st.write("Response from PaLM2 is more appropriate.")
        elif int(judge_result) == 1:
            st.write("Response from GPT-4 is more appropriate.")
