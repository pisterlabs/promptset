import streamlit as st
from streamlit_chat import message
import requests
from dotenv import load_dotenv, dotenv_values
from time import time
import json
import openai
import boto3
from time import time
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

#############################################################
# Session 내에 과거 기록이 있는지?
#############################################################
persona_prompt = """
"""


#############################################################
# Model, Tokenizer
#############################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name_or_path = ""
    
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)

#############################################################
# Session
#############################################################

st.header("Test")

if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'past' not in st.session_state:
    st.session_state['past'] = []

#############################################################
# Session 내에 과거 기록이 있는지?
#############################################################
if st.session_state['past'] == []:
    session_history_exist = False
else:
    session_history_exist = True

#############################################################
# gpt_call
#############################################################
def llm_call(prompt):
    # Encode the prompt text to tensor
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    # Generate text from the model
    output = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        temperature=temperature
    )
    
    # Decode the generated text back to a string
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return generated_text


#############################################################
# Streamlit
#############################################################
if 'user_id' not in st.session_state:
    st.session_state.user_id = ""

if 'session_num' not in st.session_state:
    st.session_state.session_num = 0


with st.form('form', clear_on_submit=True):
    
    st.write(first_message)

    user_input = st.text_input(
        'Message', 
        '', 
        key='input'
        )
    submitted = st.form_submit_button('Send')


if submitted and user_input:

    print("st.session_state['past']", st.session_state['past'])

    if st.session_state['past'] == []:
        prompt = "".join([
            persona_prompt,
            "유저: " + user_input + "\n",
            "봇: "
        ])
    else:
        
        history = []
        
        for i in range(len(st.session_state['past'])):
            history.append("유저: " + st.session_state['past'][i])
            history.append("봇: " + st.session_state['generated'][i])
        
        history = "\n".join(history)
        
        prompt = "\n".join([
            history,
            "유저: " + user_input + "\n",
            "봇: "
        ])
        
    print("prompt", prompt)

    output = llm_call(prompt).replace("\n", "")

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state['generated']:
    
    for idx, i in enumerate(range(len(st.session_state['generated'])-1, -1, -1)):
        
        message(
            st.session_state['past'][i], 
            is_user=True, 
            key=str(i) + '_user'
            )
        message(
            st.session_state["generated"][i], 
            key=str(i),
            )