import azure.cognitiveservices.speech as speechsdk
import openai
import os
import streamlit as st
import time
from dotenv import load_dotenv

st.write("**Your input:**")
input = st.empty()
st.write("**ChatBot:**")
output = st.empty()

load_dotenv('voiceBot.env')

init_prompt = f"Greet the customer according to the current day and time:{time.localtime().tm_mday}/{time.localtime().tm_mon}/{time.localtime().tm_year} {time.strftime('%H:%M:%S')}"

openai.api_key = os.environ.get('api_key')
openai.api_base = os.environ.get('api_base')
openai.api_type = os.environ.get('api_type')
openai.api_version = os.environ.get('api_version')

speech_key = os.environ.get('speech_key')
service_region = os.environ.get('service_region')

speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
speech_config.speech_synthesis_voice_name = "en-US-GuyNeural"

# Creates a speech synthesizer using the default speaker as audio output.
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config)

with open(r"C:\Users\bgraziadei\OneDrive - Maxaro\Documenten\GitHub\MaxaroProjects\VoiceBot\SystemMessage.txt") as f:
    sys_message = f.read()

conversation = [{"role": "system", "content": ""}]

def reset():
    speech_synthesizer.stop_speaking_async()
    
st.button(label="Reset", on_click=reset)

def generate_response(prompt):
    conversation.append({"role": "user", "content":prompt})
    completion=openai.ChatCompletion.create(
        engine = "PvA",
        model="gpt-3.5-turbo",
        messages = conversation
    )
    
    message=completion.choices[0].message.content
    return message

init_response = generate_response(init_prompt)
output.write(init_response)
speech_synthesizer.speak_text_async(init_response).get()
conversation.append({"role": "user", "content":init_prompt})
conversation.append({"role": "assistant", "content":init_response})

def recognition():
    while True:
        result = ""
        text = ""
        input.write("*Speak now...*")
        text = speech_recognizer.recognize_once().text

        if text:
            input.write(text)
            if text == "exit." or text == "Exit.":
                break
            result = generate_response(text)
            conversation.append({"role": "assistant", "content":result})

        if result:
            output.write(result)
            speech_synthesizer.speak_text_async(result).get()


if __name__ == "__main__":
    recognition()