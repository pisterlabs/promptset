# pipenv shell
# streamlit run chatbot_streamlit.py

import streamlit as st
import pandas as pd
import openai
import myapikeys
from text_speech_utils import *
import base64
import io

openai.api_key = myapikeys.OPENAI_KEY
input_audio_filename = 'input.wav'
output_audio_filename = 'chatgpt_response.wav'
output_conversation_filename = 'ChatGPT_conversation.txt'

def play_audio(audio_file):
    audio_bytes = open(audio_file, 'rb').read()
    audio_base64 = base64.b64encode(audio_bytes).decode()
    audio_tag = f'<audio controls autoplay><source src="data:audio/wav;base64,{audio_base64}" type="audio/wav"></audio>'
    st.markdown(audio_tag, unsafe_allow_html=True)


# Initialize app
if 'messages' not in st.session_state:
    st.session_state['messages'] = [{"role": "system", "content": "You are a general assistant for the user role \"Carl\". You will be asked questions on a variety of topics. Your first response will be a detailed response. I will then ask you to perform a second summarisation of the previous response my sending the request \"summary please\". This summary should be short and not include and code."}]

st.title("AI Annie™")
sec = st.slider("Select number of seconds of recording", min_value=2, max_value=8, value=5)

# Record audio + transcribe with Whisper + get GPT3 response
if st.button('Record audio'):
    st.write("Recording...")
    record_audio(input_audio_filename, sec)

    transcription = transcribe_audio(input_audio_filename)
    st.write(f"Me: {transcription['text']}")
    st.session_state['messages'].append({"role": "system", "content": transcription['text']})

    bot = openai.ChatCompletion.create(model="gpt-4", messages=st.session_state['messages'])
    response = bot.choices[0].message.content
    st.session_state['messages'].append({"role": "assistant", "content": response})
    st.write(f"GPT: {response}")
    print(st.session_state['messages'])
    st.session_state['messages'].append({"role": "system", "content": "summary please"})
    bot = openai.ChatCompletion.create(model="gpt-4", messages=st.session_state['messages'])
    response_summary = bot.choices[0].message.content
    print(response_summary)
    save_text_as_audio(response_summary, output_audio_filename)
    play_audio(output_audio_filename)




st.download_button(label="Download conversation", 
                   data = pd.DataFrame(st.session_state['messages']).to_csv(index=False).encode('utf-8'), 
                   file_name=output_conversation_filename)
