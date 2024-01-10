import openai
import streamlit as st
openai.api_key = st.secrets['OPENAI_API_KEY']
st.title('My first chatbot 🤖')
m = [{'role': 'system','content': 'If I say hello, say world'}]
prompt = st.text_input('Enter your message')
if prompt:
    m.append({'role': 'user','content': prompt})
    completion = openai.chat.completions.create(model='gpt-3.5-turbo',
                                            messages=m)
    response = completion.choices[0].message.content
    st.write(response) # world
