import whisper
import gradio as gr
import warnings
import openai
from gtts import gTTS
from base64 import b64encode
from io import BytesIO
import requests
import os
 
warnings.filterwarnings("ignore")
 
# Use your API key to authenticate
openai.api_key = "Your_api_key"

# 기본 'base' 모델을 로드합니다.
model = whisper.load_model("small")

def asr(audio_path):
    """
    오디오를 텍스트로 변환하고 ChatGPT를 사용하여 응답을 생성하는 함수입니다.

    Parameters:
        audio_path (str): 오디오 파일의 경로 (filepath 형식)

    Returns:
        list: 변환된 텍스트와 ChatGPT 응답이 담긴 리스트 [Speech to Text, ChatGPT Output]
    """
    # 오디오를 로드하고 30초로 패딩 또는 트리밍합니다.
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    # log-Mel 스펙트로그램을 생성하고 모델과 동일한 디바이스로 이동합니다.
    mel = whisper.log_mel_spectrogram(audio).to(model.device)

    # 오디오에서 사용 언어를 감지합니다.
    _, probs = model.detect_language(mel)

    # 오디오를 디코딩합니다.
    options = whisper.DecodingOptions(language="korean", fp16=False)
    result = whisper.decode(model, mel, options)
    result_text = result.text
    return result_text

def ask_gpt3(question, chat_log=None):
    result = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {"role": "user", "content": question}, 
        ],
    )
    response = result.choices[0].message.content
    return response, question

def tts(text: str, language: str) -> object:
    """Converts text into audio object.
    Args:
        text (str): generated answer of bot
    Returns:
        object: text to speech object
    """
    return gTTS(text=text, lang=language, slow=False)

def tts_to_bytesio(tts_object: object) -> bytes:
    """Converts tts object to bytes.
    Args:
        tts_object (object): audio object obtained from gtts
    Returns:
        bytes: audio bytes
    """
    bytes_object = BytesIO()
    tts_object.write_to_fp(bytes_object)
    bytes_object.seek(0)
    return bytes_object.getvalue()


def html_audio_autoplay(bytes: bytes) -> object:
    """Creates html object for autoplaying audio at gradio app.
    Args:
        bytes (bytes): audio bytes
    Returns:
        object: html object that provides audio autoplaying
    """
    b64 = b64encode(bytes).decode()
    html = f"""
    <audio controls autoplay>
    <source src="data:audio/wav;base64,{b64}" type="audio/wav">
    </audio>
    """
    return html


def main(audio:object, chat_log = None):
    user_message = asr(audio)
    response, question = ask_gpt3(user_message)
    bot_voice = tts(response.strip(), "ko")
    bot_voice_bytes = tts_to_bytesio(bot_voice)
    html = html_audio_autoplay(bot_voice_bytes)
    return user_message, response, html
    
    
    

# Gradio 인터페이스를 생성합니다.
output_1 = gr.Textbox(label="음성 인식 결과")
output_2 = gr.Textbox(label="ChatGPT 응답")

chat_log = None

gr.Interface(
    title = "test",
    fn=main,
    inputs=[
        gr.Audio(
            source="microphone",
            type="filepath",
        ),
    ],
    outputs=[
        gr.Textbox(label="You said: "),
        gr.Textbox(label="AI said: "),
        "html",
    ],
    live=True,
    allow_flagging="never",
).launch()
