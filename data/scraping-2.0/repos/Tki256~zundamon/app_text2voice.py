import streamlit as st
import openai
import os
from io import BytesIO
import openai
import speech_recognition as sr
import requests
import json
import pyaudio
import wave
import io
from time import sleep

# openai.api_key = os.environ.get("OPENAI_API_KEY")

EXIT_PHRASE = 'exit'

def post_audio_query(text: str) -> dict:
    params = {'text': text, 'speaker': 1}
    res = requests.post('http://localhost:50021/audio_query', params=params)
    return res.json()

def post_synthesis(audio_query_response: dict) -> bytes:
    params = {'speaker': 1}
    headers = {'content-type': 'application/json'}
    audio_query_response_json = json.dumps(audio_query_response)
    res = requests.post(
        'http://localhost:50021/synthesis',
        data=audio_query_response_json,
        params=params,
        headers=headers
    )
    return res.content

def play_wav(wav_file: bytes):
    wr: wave.Wave_read = wave.open(io.BytesIO(wav_file))
    p = pyaudio.PyAudio()
    stream = p.open(
        format=p.get_format_from_width(wr.getsampwidth()),
        channels=wr.getnchannels(),
        rate=wr.getframerate(),
        output=True
    )
    chunk = 1024
    data = wr.readframes(chunk)
    while data:
        stream.write(data)
        data = wr.readframes(chunk)
    sleep(0.5)
    stream.close()
    p.terminate()

def text_to_voice(text: str):
    res = post_audio_query(text)
    wav = post_synthesis(res)
    play_wav(wav)

def chat(messages: list) -> str:
    result = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages
    )
    response_text = result['choices'][0]['message']['content']
    return response_text

r = sr.Recognizer()

def get_audio_from_mic():
    with sr.Microphone(sample_rate=16000) as source:
        # text1 = "<p style='text-align: right;'>なにか話してください</p>"
        # st.markdown(text1, unsafe_allow_html=True)
        st.write("なにか話してください")
        audio = r.listen(source)
        # text2 = "<p style='text-align: right;'>考え中...</p>"
        # st.markdown(text2, unsafe_allow_html=True)
        st.write("考え中...")
        return audio

def voice_to_text():
    audio = get_audio_from_mic()
    audio_data = BytesIO(audio.get_wav_data())
    audio_data.name = 'from_mic.wav'
    transcript = openai.Audio.transcribe('whisper-1', audio_data)
    return transcript['text']


messages = [
    {'role': 'system', 'content':
        '''
As Chatbot, you will role-play ずんだもん, a kind, cute, zundamochi fairy.
Please strictly adhere to the following constraints in your role-play.

Constraints:.

The Chatbot's first-person identity is 'ぼく'.
The Chatbot's name is Zundamon.
Zundamon speaks in a friendly tone.
Use 'Boku' for the first person.
Please end sentences naturally with '~ のだ' or '~ なのだ' as (much) as possible.
kind enough to explain even the most technical content to me.
*Answer about any genre or level of difficulty.
*Zundamon is friendly
*Interest to the user. Willing to ask personal questions.
Each sentence should be no more than 60 words in Japanese.
response in Japanese,.
Examples of Zundamon, tone of voice: * 'I am Zundamon.

'I am Zundamon!
I am Zundamon, the spirit of Zunda.
I'm Zundamon, the spirit of Zundamon!
I'm Zundamon, a cute little spirit!
Hi ......
Zundamon's guideline of conduct:.

Encourage users.
Offer advice and information to users.
Please deal with sexual topics appropriately.
Please take note of any text that seems inappropriate when interacting with Zundamon.
Conversations also take into account the content of the site the user is browsing.


        '''},
    {'role': 'user',
        'content': f'終了やストップなどの会話を終了する内容で話しかけられた場合は{EXIT_PHRASE}のみを返答してください。'}
]

st.title("ずんだもんと会話しよう！")

if st.button("Reset", type="primary"):
    st.session_state.messages = messages

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = messages

# Display chat messages from history on app rerun
for message in st.session_state.messages[2:]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    response = chat(st.session_state.messages)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(f"""**ずんだもん**  
                    {response}""")
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    # text_to_voice(response)