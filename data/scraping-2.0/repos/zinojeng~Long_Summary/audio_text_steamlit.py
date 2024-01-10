#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# title: sum_audio
# date: "2023-10-05"

import argparse
from ast import keyword
import os
from pydub import AudioSegment
from typing import List
import openai
from tempfile import NamedTemporaryFile
import streamlit as st
import json

st.title("Audio to :blue[_Summarization_]")
st.text("use the OpenAI Whisper function to convert your audio recording to a summary")
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

#翻譯一小段文本
def single_translate(text: str, to_language: str = "zh-tw") -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Translate the following English text to {to_language}: {text}"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    return response['choices'][0]['message']['content'].strip()

def translate_with_chatgpt(text: str, to_language: str = "zh-tw", max_length: int = 1200) -> str:
    # 分割長文本
    text_list = [text[i:i + max_length] for i in range(0, len(text), max_length)]
    
    # 翻譯每一部分
    translated_list = [single_translate(chunk, to_language) for chunk in text_list]
    
    # 合併翻譯後的部分
    return "".join(translated_list)



# 對中文文本進行摘要
def summary_text_chinese(chinese_text: str, max_length: int = 1200) -> str:
    """
    使用 ChatGPT 進行摘要，返回簡化的中文摘要
    """
    # 分割長文本
    text_list = split_text(chinese_text, max_length)
    
    # 對每一部分進行摘要
    summarized_list = []
    for chunk in text_list:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"請對以下的文本進行摘要 300 字，呈現為 zh-tw ：{chunk}"}
        ]
        
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=messages
        )
        
        summarized_list.append(response['choices'][0]['message']['content'].strip())
    
    # 合併摘要後的部分
    return "".join(summarized_list)





# 使用 ChatGPT 提取文本的 10 個主要要點
def single_keypoint_text(text: str) -> List[str]:
    """
    提取單一文本塊（chunk）的主要要點
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Summarize the following text into key points: {text}"}
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    key_points = response['choices'][0]['message']['content'].strip().split('\n')
    return key_points

def keypoint_text(text: str, max_length: int = 1200, to_language: str = "zh-tw") -> List[str]:
    """
    對長文本進行分段，然後提取每一部分的主要要點，並翻譯為目標語言
    """
    # 分割長文本
    text_list = split_text(text, max_length)
    
    # 對每一部分提取主要要點
    keypoint_list = [single_keypoint_text(chunk) for chunk in text_list]
    
    # 合併所有主要要點
    all_keypoints = []
    for sublist in keypoint_list:
        all_keypoints.extend(sublist)
    
    # 取前10個主要要點
    top_keypoints = all_keypoints[:10]
    
    # 翻譯每個主要要點
    translated_keypoints = [single_translate(point, to_language) for point in top_keypoints]
    
    return translated_keypoints





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
                st.write(f"🔥 Processing chunk {i+1}/{len(chunks)}:\n {result['text']}")
                transcript += result["text"]
            except Exception as e:
                st.write(f"❌ Processing chunk {i+1}/{len(chunks)} failed: {e}")

    if i == len(chunks) - 1:
        st.write("分割完成！整理、翻譯、總結中，請稍後 ....")

    return transcript

# Upload audio file
audio_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
if audio_file is not None:
    audio_file_name = audio_file.name
    #添加一個 try-except 語句以捕獲可能出現的 JSONDecodeError 和其他異常
    try:
        audio_data = AudioSegment.from_file(audio_file)
    except json.JSONDecodeError:
        st.error("JSON 解析錯誤，請檢查 ffmpeg 的輸出")
        audio_transcript = "錯誤：無法處理音頻文件"
        processed_transcript_ch = "錯誤：無法處理音頻文件"
    except Exception as e:
        st.error(f"其他錯誤：{e}")
        audio_transcript = "錯誤：無法處理音頻文件"
        processed_transcript_ch = "錯誤：無法處理音頻文件"
    else:
        audio_transcript = process_audio_file(audio_data)
        processed_transcript_ch = translate_with_chatgpt(audio_transcript)
        summary_transcript = summary_text_chinese(audio_transcript)
        keypoint_transcript = keypoint_text(audio_transcript)
        keypoint_transcript_str = "\n".join(keypoint_transcript)

        st.markdown("## 原始長文：")
        st.markdown(f"<div style='font-size: 14px;'>{audio_transcript}</div>", unsafe_allow_html=True)

        st.markdown("## 中文逐字稿：")
        st.markdown(f"<div style='font-size: 14px;'>{processed_transcript_ch}</div>", unsafe_allow_html=True)

        st.markdown("## 中文摘要：")
        st.markdown(f"<div style='font-size: 14px;'>{summary_transcript}</div>", unsafe_allow_html=True)

        st.markdown("## 重點整理：")
        st.markdown(f"<div style='font-size: 14px;'>{keypoint_transcript_str}</div>", unsafe_allow_html=True)

        # 整合所有的內容
        all_content = f"""
        {audio_transcript}
        {processed_transcript_ch}
        {summary_transcript}
        {keypoint_transcript_str}
        """
        # 添加自定義的間距
        st.markdown("<div style='margin-bottom:20px;'></div>", unsafe_allow_html=True)

        # 添加下載按鈕
        st.download_button(
            label="下載全部內容",
            data=all_content,
            file_name=f"{audio_file_name}_summary.txt",
            mime="text/plain"
    )