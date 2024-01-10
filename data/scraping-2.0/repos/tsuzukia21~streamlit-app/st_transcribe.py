import openai
import streamlit as st
from typing import Any
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter

class JapaneseCharacterTextSplitter(RecursiveCharacterTextSplitter):
    def __init__(self, **kwargs: Any):
        separators = ["\n\n", "\n", "。", "、", " ", ""]
        super().__init__(separators=separators, **kwargs)

japanese_spliter = JapaneseCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=0,
)

with st.sidebar:
    openai.api_key = st.text_input("OpenAI API Key", key="chatbot_api_key", type="password")

st.header('音声文字起こしアプリ')

upload_file = st.file_uploader('音声文字起こしするファイルを選択してください  \nAPIの上限により25MB以上のファイルは文字起こし不可です。\
                               ファイルを分割する等容量を少なくしてください', type=['m4a', 'mp3', 'webm', 'mp4', 'mpga', 'wav'])
if upload_file is not None:
    st.subheader('ファイル詳細')
    file_details = {'FileName': upload_file.name, 'FileType': upload_file.type, 'FileSize': upload_file.size}
    st.write(file_details)
    file_name=upload_file.name.split('.')[0]
    if upload_file.size > 25000000:
        st.error('エラー：ファイルが25MBを超えています。APIの上限により25MB以上のファイルは文字起こし不可です。ファイルを分割する等容量を少なくしてください', icon="🚨")
    trans_start=st.button('文字起こし開始')

    if trans_start:
        if not openai.api_key:
            st.error('OpenAI API keyを入力してください。', icon="🚨")
            st.stop()
        else:
            with st.spinner('***音声文字起こしを実行中です...***'):
                trans= openai.Audio.transcribe("whisper-1" ,upload_file)["text"]
            st.success('***音声文字起こしを完了しました***')
            st.write("***文字起こし結果***")
            st.write(trans)
            with st.spinner('***日本語の修正中です...***'):
                texts = japanese_spliter.split_text(trans)
                texts_modified=""
                for text in texts:            
                    prompt=f"##音声文字起こしで不自然な文を削除し、自然な文章に修正してください。\n##音声文字起こし\n{text}\n##修正した文章\n"
                    messages=[{"role": "system", "content": "あなたは優秀な日本語のエディターです。"},
                        {"role": "user", "content": prompt}
                    ]
                    text_modified = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=messages,
                        temperature=0,
                    )
                    texts_modified=texts_modified+text_modified["choices"][0]["message"]["content"]
            st.success('***日本語の修正を完了しました***')
            st.write("***文字起こし結果(修正後)***")
            st.write(texts_modified)