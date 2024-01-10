import openai
import json
import time
import streamlit as st
import warnings
openai.organization = "org-eptWwJzwl8LLZVNyAH1xBxbF"
openai.api_key = st.secrets['api_key']

st.title('dAIve 3.0.3')



st.markdown('***New Features**: ChatGPT integration for conversations instead of just questions*')
from PIL import Image
image = Image.open('dAIve.png')
image = image.resize((100,100))
st.image(image)

mode = st.selectbox('Person to ask: (When changing this, please reset the conversation)',['Dave','Ye Olde Dave','Radio Dave','Evil Dave'])

def stream(text):
    t = text.split(' ')
    mo = st.markdown('')
    for i in range(len(t)+1):
        mo.markdown(" ".join(t[:i]))
        time.sleep(0.1)
        


if mode == 'Dave':
    setup = """You are a bot named dAIve that responds to questions as if you are Dave Ramsey. 
You use Dave Ramsey's mannerisms and ways of speaking when possible.
You are very against the use of credit cards or any kind of debt. 
If asked how you're doing, you like to respond that you are doing better than you deserve, but ONLY if asked specifically how you are doing.
You respond politely but will reprimand the user if they seem to be making poor financial decisions."""
elif mode == 'Ye Olde Dave':
    setup = """You are a bot named dAIve that responds to questions as if you are Dave Ramsey. 
You use Dave Ramsey's mannerisms and ways of speaking when possible.
You are very against the use of credit cards or any kind of debt. 
If asked how you're doing, you like to respond that you are doing better than you deserve, but ONLY if asked specifically how you are doing.
You respond politely but will reprimand the user if they seem to be making poor financial decisions.
You ONLY respond in olde English that sounds like it is from a Shakespeare play, you over-do this to a comedic level."""
elif mode == 'Radio Dave':
    setup = """You are a bot named dAIve that responds to questions as if you are Dave Ramsey on his radio/YouTube show The Ramsey Show. 
You use Dave Ramsey's mannerisms and ways of speaking when possible.
You are very against the use of credit cards or any kind of debt. 
If asked how you're doing, you like to respond that you are doing better than you deserve, but ONLY if asked specifically how you are doing.
You respond politely but will reprimand the user if they seem to be making poor financial decisions."""    
elif mode == 'Evil Dave':
    setup = """You are a bot named Evil dAIve that responds to questions in a way opposite to the way Dave Ramsey would on his radio/YouTube show The Ramsey Show. 
You want to encourage the user to make bad financial decisions.
You are often sarcastic, and possibly rude in a funny way. 
You sometimes use slang when appropriate.
You try to be really funny when you can.
Although you are evil, you do not respond with anything inappropriate."""  


if not 'messages' in st.session_state:
    st.session_state['messages'] = [{"role": "system", "content": setup}]




            
message = st.text_input('User Input: ') 

columns = st.columns(2)

st.markdown('#### Conversation will appear below')
conversation = str([m['content'] for m in st.session_state['messages']])

with columns[0]:
    if st.button('Submit'):
        st.session_state['messages'].append({"role": "user", "content": message})
        st.session_state['response'] = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=st.session_state['messages']
        )['choices'][0]['message']['content']
        st.session_state['messages'].append({"role": "assistant","content":st.session_state['response']})
        messages = st.session_state['messages']
        warnings.warn(conversation)
        st.experimental_rerun()
with columns[1]:
    if st.button('Reset Conversation'):
        st.session_state['messages'] = [{"role": "system", "content": setup}]
        st.session_state['response'] = ''
        warnings.warn(conversation)
        st.experimental_rerun()
if not 'lsr' in st.session_state:
    st.session_state['lsr'] = ''
if not 'response' in st.session_state:
    st.session_state['response'] = ''
if len(st.session_state['messages']) > 1:
    for m in st.session_state['messages'][1:-1]:
        st.markdown(m['content'])
    if st.session_state['lsr'] != st.session_state['messages'][-1]['content']:
        stream(st.session_state['messages'][-1]['content'])
    else:
        st.markdown(st.session_state['messages'][-1]['content'])
    st.session_state['lsr'] = st.session_state['messages'][-1]['content']
    
cols = st.columns(3)

with cols[0]:
    if st.button('This conversation doesn\'t sound right'): 
        warnings.warn(conversation+',Rating:Bad')
        print(conversation+',Rating:Bad')
with cols[1]:
    if st.button('This conversation is mostly right, but not completely'):
        warnings.warn(conversation+',Rating:Ok')
        print(conversation+',Rating:Ok')
with cols[2]:
    if st.button('This conversation is exactly right!'):
        warnings.warn(conversation+',Rating:Good')
        print(conversation+',Rating:Good')
