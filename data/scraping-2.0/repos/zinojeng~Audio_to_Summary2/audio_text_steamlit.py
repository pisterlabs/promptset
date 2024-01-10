#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# title: sum_audio
# date: "2023-10-05"

import argparse
import os
from pydub import AudioSegment
from typing import List
import openai
from tempfile import NamedTemporaryFile
import streamlit as st

st.markdown("<h1 style='text-align: center; color: blue;'>Audio to Summarization</h1>", unsafe_allow_html=True)

st.text("Utilize the OpenAI Whisper function to convert your audio recording to a summary")
st.text("use gpt-3.5-turbo-16k: faster and significantly cheaper to run")

# 獲取API金鑰，從環境變數而非硬編碼
# User can input OpenAI API Key
api_key = st.text_input(
      label="Enter your OpenAI API Key:", 
      placeholder="Ex: sk-2twmA8tfCb8un4...", 
      key="openai_api_key", 
      help="You can get your API key from https://platform.openai.com/account/api-keys/")
if api_key:
    openai.api_key = api_key

# Get system role message from the user
system_prompt = st.text_input('Enter a system role message:')
st.caption("Example: You specialize in endocrinology and diabetes....")


# 文字分割
def split_text(text: str, max_length: int) -> List[str]:
    """
    將文字分割為指定最大長度的子字符串
    """
    return [text[i:i + max_length] for i in range(0, len(text), max_length)]

def process_long_text(long_text, type):
    text_list = split_text(long_text, 1200)
    processed_text_list = [process_text(text, type) for text in text_list]
    return "".join(processed_text_list)

# 處理文字
def process_text(text: str, type: str) -> str:
    """
    使用 ChatGPT 處理文字，返回總結
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Please summarize the following text in 500 words as detail as you can in zh-tw: {text}"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response['choices'][0]['message']['content'].strip()

# 使用 ChatGPT 進行翻譯
def translate_with_chatgpt(text: str, to_language: str = "zh-tw") -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Translate the following English text to {to_language}: {text}"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response['choices'][0]['message']['content'].strip()
def translate_long_text(text: str, to_language: str = "zh-tw", max_length: int = 1200) -> str:
    # 分割長文本
    text_list = [text[i:i + max_length] for i in range(0, len(text), max_length)]
    
    # 翻譯每一部分
    translated_list = [translate_with_chatgpt(chunk, to_language) for chunk in text_list]
    
    # 合併翻譯後的部分
    return "".join(translated_list)


# 使用 ChatGPT 提取文本的 10 個主要要點
def summarize_with_chatgpt(text: str) -> List[str]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Summarize the following text into 10 key points in zh-tw: {text}"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    key_points = response['choices'][0]['message']['content'].strip().split('\n')
    return key_points[:10]


def process_audio_file(audio_data):
    st.write("讀入檔案📂")
    st.write("切割音檔成多個小檔案中，請稍後...📚")

    chunk_size = 100 * 1000  # 100 秒
    chunks = [
        audio_data[i:i + chunk_size]
        for i in range(0, len(audio_data), chunk_size)
    ]

    openai.api_key = api_key

    transcript = ""
    for i, chunk in enumerate(chunks):
        with NamedTemporaryFile(suffix=".wav", delete=True) as f:
            chunk.export(f.name, format="wav")
            try:
                result = openai.Audio.transcribe(
                    "whisper-1",
                    f,
                    prompt="I’ll be having an English language conversation on a topic you might find quite interesting.",
                    options={
                        "language": "en",
                        "temperature": "0"
                    }
                )
                #st.write(f"🔥 Processing chunk {i+1}/{len(chunks)}:\n {result['text']}")
                transcript += result["text"]
            except Exception as e:
                # 當異常發生時執行此塊
                st.write(f"錯誤：{e}")

    if i == len(chunks) - 1:
        st.write("分割完成！整理、翻譯、總結中，請稍後 ....")


    # 使用 ChatGPT 進行後處理（這裡僅為示例）
    processed_transcript = process_text(transcript, "proofread")
    processed_transcript_ch = translate_long_text(transcript)
    processed_summary = summarize_with_chatgpt(processed_transcript)
    processed_summary_str = "\n".join(processed_summary)

    st.markdown("## 原始長文：")
    st.markdown(f"<div style='font-size: 14px;'>{transcript}</div>", unsafe_allow_html=True)

    st.markdown("## 中文逐字稿：")
    st.markdown(f"<div style='font-size: 14px;'>{processed_transcript_ch}</div>", unsafe_allow_html=True)

    st.markdown("## 中文摘要：")
    st.markdown(f"<div style='font-size: 14px;'>{processed_transcript}</div>", unsafe_allow_html=True)

    st.markdown("## 重點整理：")
    st.markdown(f"<div style='font-size: 14px;'>{processed_summary_str}</div>", unsafe_allow_html=True)

 
# Upload audio file
audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
if audio_file is not None:
    audio_data = AudioSegment.from_file(audio_file)
    process_audio_file(audio_data)
#else:
#    st.write("No audio file uploaded or audio_file is None.")
