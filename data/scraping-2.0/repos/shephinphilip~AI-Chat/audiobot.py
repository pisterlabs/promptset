import streamlit as st
import openai
import speech_recognition as sr
from gtts import gTTS
 
# Set your OpenAI API key
openai.api_key = ""
 
# Streamlit UI Improvements with Custom Style
custom_style = """
<style>
body, html, .stApp {
    background-image: linear-gradient(to right, #6DD5FA, #FF758C);
    color: #FFFFFF;
}
 
h1 {
    color: #FFFFFF;
}
 
.stButton>button {
    color: #FFFFFF;
    background-color: #FF4B2B;
    border-radius: 5px;
    padding: 10px 24px;
    font-size: 16px;
    font-weight: bold;
    margin: 10px 0px;
}
 
.stTextInput, .stSelectbox, .stTextarea {
    border-radius: 5px;
}
 
.css-1cpxqw2 {
    background-color: #292929;
    border-radius: 10px;
    padding: 10px;
    color: #FFFFFF;
}
 
.css-1cpxqw2:hover {
    background-color: #3d3d3d;
}
 
</style>
"""
st.markdown(custom_style, unsafe_allow_html=True)
 
# Header Section
col1, col2 = st.columns([1, 3])
with col1:
    st.image('https://img1.wsimg.com/isteam/ip/8ae99362-3f93-4e87-9993-16961c92132c/omnivalue_450x450-nobg.png/:/rs=w:52,h:52,cg=true,m/cr=w:52,h:52/qt=q:95', width=100)
with col2:
    st.title("Welcome to Omni Value Solutions")
    st.write("Type your message or use the 🎙️ button to start recording your query. Wait for the response and listen to the playback.")
 
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
 
# Function to generate OpenAI response and convert text to speech
def generate_response_and_speak(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    text_response = response['choices'][0]['message']['content'].strip()
    response_audio = text_to_speech(text_response)
    return text_response, response_audio


 
def text_to_speech(text, filename="response.mp3"):
    tts = gTTS(text=text, lang='en')
    tts.save(filename)
    return filename
 
# Initialize speech recognition
recognizer = sr.Recognizer()
 
# Process text input or voice input
def process_input(input_text, is_voice=False):
    if "@audiobot" in input_text or is_voice:
        openai_response, response_audio = generate_response_and_speak(input_text)
        st.session_state.messages.append({"role": "assistant", "content": openai_response})
        if is_voice:
            st.audio(response_audio, format="audio/mp3")
 
# Chat and Voice Recording UI
st.write("Type your message or record your voice.")
col1, col2 = st.columns([3, 1])
 
with col1:
    user_input = st.text_input("Type your message here...", key="text_input")
 
with col2:
    if st.button("🎙️ Record", key="record_button"):
        with sr.Microphone() as source:
            st.write("Recording... Speak now.")
            audio_input = recognizer.listen(source, phrase_time_limit=5)
            st.write("Recording complete. Processing...")
 
            try:
                audio_text = recognizer.recognize_google(audio_input)
                st.session_state.messages.append({"role": "user", "content": audio_text})
                process_input(audio_text, is_voice=True)
            except sr.UnknownValueError:
                st.error("Could not understand the audio.")
            except sr.RequestError as e:
                st.error(f"Could not request results; {e}")
 
# Send button for text input
if st.button("Send", key="send_button"):
    user_message = st.session_state.text_input
    st.session_state.messages.append({"role": "user", "content": user_message})
    process_input(user_message)
 
# Display chat history
for message in st.session_state.messages:
    if message["role"] == "user":
        st.container().markdown(f"<div style='text-align: left; background-color:#007bff; color: white; padding:10px; border-radius:10px; margin-bottom:5px;'>You: {message['content']}</div>", unsafe_allow_html=True)
    elif message["role"] == "assistant":
        st.container().markdown(f"<div style='text-align: left; background-color:#28a745; color: white; padding:10px; border-radius:10px; margin-bottom:5px;'>Assistant: {message['content']}</div>", unsafe_allow_html=True)
 
# Footer
st.markdown("---")
st.text("Omni Value Solutions © 2023")
 
