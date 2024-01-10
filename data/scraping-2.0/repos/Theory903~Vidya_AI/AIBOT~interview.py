import os
import openai
import streamlit as st

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.getenv("OPENAI_API_KEY")

st.title("Interview Preparation")
st.markdown(f"<style>.st-emotion-cache-k7vsyb span  {{font-family: bold; font-size: 48px;color:#d1e2f7;text-align:center;text-shadow: 4px 5px 5px #4c5662 }}</style>", unsafe_allow_html=True)


st.markdown(f"<style>.st-emotion-cache-ffhzg2 {{background-image: url('https://images.pexels.com/photos/6044198/pexels-photo-6044198.jpeg?auto=compress&cs=tinysrgb&w=1260&h=750&dpr=2');background-size: cover; border: 5px solid #252222;border-radius:60px  }}</style>", unsafe_allow_html=True)

text_input = st.text_input("how can i help you",placeholder = "talk with me", key = "input")

role = st.selectbox("ROLE",index = None, options = ("user","developer"))


if role and text_input:
    
    message = {}
    
    message["role"] = role
    message["content"] = "generate interview Question"+text_input
    
    response = openai.ChatCompletion.create(
        
        model="gpt-3.5-turbo",
        messages=[message],
        temperature=0.5,
        max_tokens=1024)


    if "history" not in st.session_state:
        st.session_state["history"] = []
        
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["Hello how can i help you"]
        
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey ! :wave: "]
        
        
    #response_container = st.container()
    
    container = st.container()
    
    with container:
        st.title("Thanks for asking")
        st.write(response["choices"][0]["message"]["content"])
        

