# Import necessary libraries
import streamlit as st
import os 
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')

# Define function to generate bot response using OpenAI API
def generate_response(prompt):
    completion = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0.5,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    response = completion.choices[0].text
    return response

# Define function to print messages to Streamlit app
def message(text, is_user=False, key=None):
    if is_user:
        st.write("You: " + text, key=key)
    else:
        st.write("Bot: " + text, key=key)

# Set Streamlit app configuration
st.set_page_config(page_title="Mohsin's Bot", page_icon=":robot_face:", layout="wide", initial_sidebar_state="collapsed")

# Add title and description to the Streamlit app
st.markdown("<h1 style='text-align: center; color: white;'>Welcome to Mohsin's Bot!</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center; color: white;'>Ask me anything and I'll do my best to help you out.</h4>", unsafe_allow_html=True)

# Create session state variables for the bot's past responses and generated responses
if 'generated' not in st.session_state:
    st.session_state['generated'] = ['I am ready to help you, sir.']

if 'past' not in st.session_state:
    st.session_state['past'] = ['Hey there!']

# Define function to get user input from text box
def get_text():
    input_text = st.text_input("", key="input")
    return input_text

# Get user input and generate bot response if input is detected
user_input = get_text()
if user_input:
    output = generate_response('\n'.join(st.session_state.past[-3:] + [user_input]))
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

# Print generated and past responses to the Streamlit app
if st.session_state['generated']:
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')

# Add footer to Streamlit app
st.markdown("<p style='text-align: center; color: white;'>Made with :heart: by Mohsin</p>", unsafe_allow_html=True)

# Set the background color of the Streamlit app to black
st.markdown("""
<style>
body {
    background-color: #000000;
}
</style>
""", unsafe_allow_html=True)
