import streamlit as st
from openai import OpenAI
import consts
import os
import tempfile

# Initialize chat history in session state
if 'audio_history' not in st.session_state:
  st.session_state.audio_history = []
# Placeholder for chat messages
chat_container = st.empty()
os.environ['OPENAI_API_KEY'] = consts.API_KEY_OPEN_AI
client = OpenAI()

def clear_history():
  st.session_state.audio_history = []
  chat_container.empty()  # Clear the chat input box

def show_audio(message): 
  # if message is not empty, do show audio
  if message != '':
    speech_file_path = tempfile.NamedTemporaryFile(delete=True).name
    with st.spinner('AI is preparing...'):
      response = client.audio.speech.create(
        model="tts-1",
        voice=selected_voice,
        input=message
      )
      response.stream_to_file(speech_file_path)
      st.audio(speech_file_path)

def show_audio_history():
  for author, message in st.session_state.audio_history:
    with st.chat_message(author):
      if author == 'AI Assistant':
        show_audio(message)
      else:
        st.write(message)

# Streamlit UI
st.title("AI-powered Text To Speech")
voices = ['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'] 
selected_voice = st.selectbox("Choose a voice", voices)
    
# Chat input for user message
user_message = st.chat_input("Type something here for AI to read...")

if user_message:
  clear_history() 
  # Add user message to chat history
  st.session_state.audio_history.append(('user', user_message))
  st.session_state.audio_history.append(('AI Assistant', user_message))
  show_audio_history()  

# Clear chat history button
if len(st.session_state.audio_history) >= 2:
  if st.button("Clear this speech"):
    clear_history()  # Clear the chat history
    st.experimental_rerun()





