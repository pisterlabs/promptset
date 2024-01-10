import streamlit as st
import os
from openai import OpenAI

OpenAI.api_key = os.environ["OPENAI_API_KEY"]

client = OpenAI()

input_text = st.text_input("作成したいプログラムの内容を入力してください")

if st.button("実行"):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        temperature=0,
        messages=[
            {"role": "system", "content": "あなたは優れたプロの日本人プログラマーです"},
            {"role": "user", "content": input_text},
        ],
        max_tokens=1000,
    )
    st.write(completion.choices[0].message.content)
