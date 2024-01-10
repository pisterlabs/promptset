import os
import streamlit as st
import pdfplumber
import openai

# 设置你的 OpenAI 密钥
openai.api_key = os.getenv("API_KEY")
st.header("虚拟面试官")
st.subheader( "根据上传的简历来分析您的优势")
st.subheader( "同时为您提供面试题，来模拟面试")

def read_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
    return text

def summarize_text(text):
    messages = [
            {"role": "system", "content": "你现在来辅助帮助用户发现用户简历中的优势，和你认为可以优化的内容，语言干练简洁"},
            {"role": "user", "content": text},
        ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages
    )
    return response['choices'][0]['message']['content']
def summarize_question(text):
    messages = [
            {"role": "system", "content": "你现在扮演面试官，阅读分隔符 #### 之间的用户简历,按照用户的专业给用户出几个专业度高的面试题,####"+text+"#### "},
           #{"role": "user", "content": text},
        ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages
    )
    return response['choices'][0]['message']['content']
if 'summarized' not in st.session_state:
    st.session_state.summarized = False
    st.session_state.summary = None
    st.session_state.text = None
uploaded_file = st.file_uploader("上传pdf格式的简历", type='pdf')

if uploaded_file is not None and not st.session_state.summarized: 
    with st.spinner('正在读取PDF...'):

        st.session_state.text = read_pdf(uploaded_file)
    with st.spinner('正在挖掘闪光点并提供一些优化建议...'):
        st.session_state.summary = summarize_text(st.session_state.text)
        st.session_state.summarized = True
        st.write( st.session_state.summary)
        # 添加一个按钮
if st.session_state.summarized:
    if st.button('我是面试官，我来提出几个问题吧！'):
        st.write("")
        with st.spinner('正在准备实战题目...'):
            st.session_state.summary = summarize_question(st.session_state.text)
        st.write(st.session_state.summary)