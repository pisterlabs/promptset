import streamlit as st
# text to speech
from gtts import gTTS 
# playing speech
from playsound import playsound
# GPT calls
import openai
# transcribe audio
import whisper

# record audio
from audiorecorder import audiorecorder


openai.organization = st.secrets['OPENAI_ORG']
openai.api_key = st.secrets['OPENAI_API_KEY']

user_message = 'enter an input to receive an output'
input_text =''
audio_input_text=''
bot_message = "no chat sent yet! click \"send message \" to chat."
ipsum = """Lorem ipsum dolor sit amet, 
consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. 
Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo 
consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat 
nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt
 mollit anim id est laborum."""


### NOTE: Process taken
### 1) take in audio from the user based on button press
### 2) transcribe audio to text and store
### 3) tell user audio was taken in; send it back to them
### 4) send this text to GPT to get response back
### 5) add button to show text of the response
### 6) send back audio of the response


def record_audio():
    audio = audiorecorder("Record audio", "Stop recording")
    if len(audio) > 0:
    # To play audio in frontend:
        st.audio(audio.export().read())
        audio.export("audio.wav", format="wav")
        return "audio.wav"
    return ""
 
# Whisper performs speech-to-text
# give it output.wav
def audio_to_text(audio_file):
    model = whisper.load_model("small")
    return model.transcribe(audio_file)['text']

# takes in prompt and sends back response
def call_turbo(prompt, max_tokens=1000):
    response = openai.ChatCompletion.create(
    model = 'gpt-3.5-turbo',
    messages = [
        {"role": "user", "content": prompt}
    ],
    max_tokens = max_tokens
    )
    # parse for text
    return response['choices'][0]['message']['content']


def toSpeech(message, language):
    speech = gTTS(text = message, lang=language)
    speech.save('example.mp3')
    playsound('example.mp3')

##
##    Beginning of
##    App Frontend
##

st.title("Conversational GPT")
left, right = st.columns(2, gap = "medium", )

with left: 
    st.write("tap record to send audio:")
    audio_input = record_audio()
    if len(audio_input) > 0:
        audio_input_text = audio_to_text(audio_input)
    st.write("")
    input_text = st.text_area(label="chat below without audio:", height=10)
    if st.button("send message"):
        if len(input_text)>0:
            bot_message = call_turbo(input_text)
        elif audio_input_text:
            bot_message = call_turbo(audio_input_text)
        else:
            pass
    if audio_input_text:
        st.write(f"You said:\n{audio_input_text}")

with right:
    st.subheader("GPT")
    st.write(bot_message)