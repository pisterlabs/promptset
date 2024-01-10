import uuid

import streamlit as st
from st_audiorec import st_audiorec
from openai import OpenAI
from pydub import AudioSegment
from libs.msal import msal_auth
from libs import get_data_dir
import io
import sys
import os
from dotenv import load_dotenv

from libs.session import PageSessionState

sys.path.append(os.path.abspath('..'))
load_dotenv()

page_state = PageSessionState("speech")

# 用于存储临时文件
data_dir = get_data_dir()

page_state.initn_attr("input_type", "microphone")
page_state.initn_attr("audio_text_source", None)
# 语音合成内容
page_state.initn_attr("speech_recode", None)

st.sidebar.markdown("# 🎙️语音转录🎤")

# 原始语音文本，识别或者上传的内容
content_box = st.empty()

if page_state.audio_text_source is not None:
    content_box.markdown(page_state.audio_text_source)
else:
    content_box.empty()


def on_flie_change():
    if page_state.uploaded_file is not None:
        stringio = io.StringIO(page_state.uploaded_file.getvalue().decode("utf-8"))
        string_data = stringio.read()
        page_state.audio_text_source = string_data
        content_box.markdown(string_data)


def clear_result():
    page_state.audio_text_source = None
    page_state.speech_recode = None
    content_box.empty()


if page_state.audio_text_source is None:
    wav_audio_recode = st_audiorec()
    if wav_audio_recode is not None:
        with st.spinner('正在识别语音...'):
            audio_segment = AudioSegment.from_wav(io.BytesIO(wav_audio_recode))
            filename = os.path.join(data_dir, f"{uuid.uuid4()}.audio.wav")
            audio_segment.export(filename, format="wav")
            client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                response_format="json",
                file=open(filename, "rb"),
            )
            page_state.audio_text_source = transcript.text
            content_box.markdown(page_state.audio_text_source)

st.sidebar.file_uploader("上传文本文件", type=["txt", "md"],
                         on_change=on_flie_change, key="speech_uploaded_file")

st.sidebar.button("清空数据", on_click=clear_result)

# 是否已经识别语音保存文本结果， 如果有就展示合成语音界面部分
if page_state.audio_text_source is not None:
    sound = st.selectbox("选择音色", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
    c1, c2, c3 = st.columns(3)
    if c1.button("合成语音"):
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        speech_file_path = os.path.join(data_dir, f"{uuid.uuid4()}.speech.mp3")
        with st.status("正在合成语音", expanded=True) as status:
            response = client.audio.speech.create(
                model="tts-1",
                voice=sound,
                input=page_state.audio_text_source
            )
            page_state.speech_recode = response.read()
            status.update(label="语音合成完成!", state="complete")

    if page_state.speech_recode is not None:
        st.write(f"🎧{sound}音色")
        st.audio(page_state.speech_recode, format="audio/mp3")
        st.write(f"语音{sound}合成完成")
        c3.download_button(
            label="下载语音",
            data=page_state.speech_recode,
            file_name='speech.mp3',
        )
