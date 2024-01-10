# -*- coding: utf-8 -*-
"""
Created on Wed Dec 28 19:44:16 2022

@author: Pravitha
"""
import openai
import streamlit as st
#import streamlit_chat
from streamlit_chat import message
import json
from PIL import Image



openai.api_key = "Your API Key"


#background images
img=Image.open("back.jpg")
page_bg_img = f"""
<style>
[data-testid="stAppViewContainer"] > .main {{
background-image: url("https://wallpx.com/thumb/2020/11/green-gradient-simple-minimalistic-364.jpg");
background-size: cover;
background-position: left;
background-repeat: no-repeat;
background-attachment: local;
}}
[data-testid="stSidebar"] > div:first-child {{
background-image: url("data:image/png;base64,{img}");
background-position: center;
background-repeat: no-repeat;
background-attachment: fixed;
}}
[data-testid="stHeader"] {{
background: rgba(0,0,0,0);
}}
[data-testid="stToolbar"] {{
right: 2rem;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

df=json.load(open('data (1).json','rb'))


def generate_reply(df):
    completions = openai.Completion.create(
    engine ="text-davinci-003",
    prompt=df,
    max_tokens=150,
    n=1,
    stop=None,
    temperature=0.7
    )
    
    message=completions.choices[0].text
    return message
#storing for chat
if "history" not in st.session_state:
    st.session_state.history = []

st.title("Omdena-Paris AI Chatbot")

st.header("GPT-3's 'text-davinci-003 model'")
st.subheader("Conversational AI chatbot for the Elderly and Disabled")


def get_text():
    input_text=st.text_input("TALK TO THE BOT", key="input_text")
    return input_text

user_input =get_text()
if user_input:
    output=generate_reply(user_input)
    #store the output
    st.session_state.history.append({"message": user_input, "is_user": True })
    st.session_state.history.append({"message": output, "is_user": False})

    
for chat in st.session_state.history:
    message(**chat)  # unpacking
    
    
