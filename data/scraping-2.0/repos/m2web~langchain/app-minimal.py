# Import necessary modules
import os 
from apikey import apikey 
import streamlit as st 
from langchain.llms import OpenAI

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey

# Set the title of the Streamlit app
st.title('🦜🔗 Mark\'s GPT Creator')

# Create a text input for the user to enter their prompt
prompt = st.text_input('Plug in your prompt here') 

# Initialize an OpenAI language model with a temperature of 0.9
llm = OpenAI(temperature=0.9) 

# If the user has entered a prompt, generate a response using the language model
if prompt: 
    response = llm(prompt)
    st.write(response)