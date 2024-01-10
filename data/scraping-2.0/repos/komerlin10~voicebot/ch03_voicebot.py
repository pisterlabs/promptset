import streamlit as st
import openai
from audiorecorder import audiorecorder
import numpy as np
import os
from datetime import datetime
from gtts import gTTS
import base64

def STT(audio):
    filename = "input.mp3"
    wav_file = open(filename, "wb")
    wav_file.write(audio.export().read())
    wav_file.close()
    
    # 음원 파일 열기
    audio_file = open(filename, "rb")
    #whisper 모델을 활용해 텍스트 얻기
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    audio_file.close()
    #파일 삭제
    if os.path.exists(filename):
        os.remove(filename)

    return transcript["text"]

def TTS(response):
    # gTTS를 활용해 음성 파일 생성
    filename = "output.mp3"
    tts = gTTS(text=response, lang="ko")
    tts.save(filename)
    
    # 음원 파일 자동 재생
    with open(filename, "rb") as f:
        audio = f.read()
        b64 = base64.b64encode(audio).decode()
        md = f"""
            <audio autoplay="True">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
        """
        
        st.markdown(md, unsafe_allow_html=True,)
        
    # 파일 삭제
    os.remove(filename)
    

def ask_gpt(prompt, model):
    response = openai.ChatCompletion.create(model=model, messages=prompt)
    system_message = response["choices"][0]["message"]
    return system_message["content"]

def main():
    st.set_page_config(
        page_title="음성 비서 프로그램",
        layout="wide"
    )
    
    flag_start = False
    
    # session state 초기화
    if "chat" not in st.session_state:
        st.session_state["chat"] = []

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "system", "content": "You are a thoughtful assistant. Respond to all input in 25 words and answer in korea"}]

    if "check_audio" not in st.session_state:
        st.session_state["check_audio"] = []
    
    
    st.header('음성 비서 프로그램')
    
    st.markdown("---")
    
    with st.expander("음성비서 프로그램에 관하여", expanded=True):
        st.write(
        """
        - 음성비서 프로그램의 UI는 스트림릿을 활용하였습니다.
        - STT(Speech-To-Text)는 OpenAI의 Whisper AI를 활용했습니다. 
        - 답변은 OpenAI의 GPT 모델을 활용했습니다. 
        - TTS(Text-To-Speech)는 구글의 Google Translate TTS를 활용했습니다.
        """)
        
        st.markdown("")
    
    # session state 초기화
    if "chat" not in st.session_state:
        st.session_state["chat"] = []
        
    if "message" not in st.session_state:
        st.session_state["message"] = [{"role": "system", "content": "You are a thoughtful assistant. Respond to all input in 25 words and answer in korean"}]
        
    if "check_audio" not in st.session_state:
        st.session_state["check_audio"] = []
        
    
        
    with st.sidebar:
        
        openai.api_key = st.text_input(label="OpenAI API Key", placeholder="OpenAI API Key를 입력하세요.", value="", type="password")
        
        st.markdown("---")
        
        model = st.radio(label="GPT 모델 선택", options=["gpt-4", "gpt-3.5-turbo"])
        
        st.markdown("---")
        
        if st.button(label="초기화"):
            st.session_state["chat"] = []
            st.session_state["message"] = [{"role": "system", "content": "You are a thoughtful assistant. Respond to all input in 25 words and answer in korean"}]
            # st.session_state["check_audio"] = []
        
    col1, col2 = st.columns(2)
    with col1:
        # audio_file = st.file_uploader(label="음성 파일 업로드", type=["mp3", "wav"])
        st.subheader("질문하기")
        # 음성 녹음 아이콘 추가
        audio = audiorecorder("클릭하여 녹음하기", "녹음 중...")
        if not audio.empty() and not np.array_equal(audio, st.session_state["check_audio"]):
            # 음성 재생
            st.audio(audio.export().read())
            # 음원 파일에서 텍스트 추출
            question = STT(audio)
            
            #채팅을 시각화하기 위해 질문 내용 저장
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state["chat"] += [("user", now, question)]
            
            #질문 내용 저장
            st.session_state["messages"] += [{"role": "user", "content": question}]
            
            # audio 버퍼를 확인하기 위해 오디오 정보 저장
            st.session_state["check_audio"] = audio
            flag_start = True
            
    
    with col2:
        # st.audio(audio_file)
        st.subheader("질문/답변")
        if flag_start:
            #ChatGPT 에게 답변 얻기
            response = ask_gpt(st.session_state["messages"], model)
            print(f"response: {response}")
            
            # GPT 모델에 넣을 프롬프트를 위해 답변 내용 저장
            st.session_state["messages"] += [{"role": "system", "content": response}]
            
            # 채팅 시각화를 위한 답변 내용 저장
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state["chat"] += [("bot", now, response)]
            print(f"chat: {st.session_state['chat']}")
            
            # 채팅 형식으로 시각화 하기
            for sender, time, message in st.session_state["chat"]:
                if sender == "user":
                    st.write(f'<div style="display:flex;align-items:center;"><div style="background-color:#007AFF;color:white;border-radius:12px;padding:8px 12px;margin-right:8px;">{message}</div><div style="font-size:0.8rem;color:gray;">{time}</div></div>', unsafe_allow_html=True)
                    st.write("")
                else:
                    st.write(f'<div style="display:flex;align-items:center;justify-content:flex-end;"><div style="background-color:lightgray;border-radius:12px;padding:8px 12px;margin-left:8px;">{message}</div><div style="font-size:0.8rem;color:gray;">{time}</div></div>', unsafe_allow_html=True)
                    st.write("")
        
            # gTTS를 활용해 음성 파일 생성 및 재생
            TTS(response)
    
        
        
        
if __name__ == '__main__':
    main()