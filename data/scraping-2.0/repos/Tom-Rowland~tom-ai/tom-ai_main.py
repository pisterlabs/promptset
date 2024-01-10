import streamlit as st
import app_functions
import database
from data import tom_context
from config import OPENAI_TOKEN, RESEMBLEAI_TOKEN, RESEMBLEAI_PROJECTID,RESEMBLEAI_VOICEID

st.title("tom.ai")
st.text("Start by saying hello")

if 'count' not in st.session_state:
    st.session_state.count = 0

i = 0

if 'chat_history' not in st.session_state:
    st.session_state.session_key = database.write_session()
    st.session_state.chat_history = tom_context

while True:
    text = st.text_input(" ", key=f"text_input_{i}")
    
    if not text:
        break
    
    if i == st.session_state.count:
        database.write_chat(st.session_state.session_key,None,True,text)
        st.session_state.chat_history += f'\n\n[User]: {text}\n\n[Tom]: '
        del text
        # Generate response to input with ChatGPT
        response = app_functions.generate_response(st.session_state.chat_history)
        st.session_state.chat_history += str(response)
        
        # Check if phrase has already been synthesised in resemble.ai
        clip_id = app_functions.check_clip_already_exists(response)

        if not clip_id: # Create clip in resemble.ai of the response and return the new clip's uid
            clip_id, body = app_functions.create_clip(response)
            
        
        # Get audio file url from resemble.ai
        audio_file, audio_file_key = app_functions.get_clip(clip_id, response)
        database.write_chat(st.session_state.session_key,audio_file_key,False,response)
        
        # Render audio in web
        st.audio(audio_file, format='audio/wav')
        st.session_state.count += 1
    i += 1