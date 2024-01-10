import streamlit as st
import openai

openai.api_key = st.secrets["api_key"]
#"YOUR API KEY"

st.header('Tagline Generation App')
st.text('(Powered by GPT-3 Model)')

col1, col2 = st.columns([3,1])
with col1:
    prompt = st.text_area('Enter your prompt', "", height=180)    

with col2:
    temp = st.slider('Temperature', 0.0, 1.0, 1.0)
    mt = st.number_input('Maximum token', 1, 4000, 60)

trigger = st.button("Generate")
st.subheader("Result")

if trigger:
    response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=temp, max_tokens=mt)
    result = response["choices"][0]["text"].lstrip("\n")
    st.write(result)
