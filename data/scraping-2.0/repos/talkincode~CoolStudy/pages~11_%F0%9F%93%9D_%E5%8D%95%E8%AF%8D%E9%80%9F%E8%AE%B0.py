import streamlit as st
import sys
import os
from dotenv import load_dotenv
from libs.llms import openai_streaming
sys.path.append(os.path.abspath('..'))
load_dotenv()

if 'user_input' not in st.session_state:
    st.session_state.user_input = ""

if 'study_tate' not in st.session_state:
    st.session_state.study_tate = {}

# 创建侧边栏
sidebar = st.sidebar
if sidebar.button('输入单词'):
    st.markdown('输入一个单词')
user_input = st.text_input('请输入一个单词')


if st.button("开始学习"):
    with st.spinner("生成中..."):
        msg = f"""
        请按照翻译用户输入的单词或中文，如果输入中文就直接翻译为英文,一定要有中文的提示
        输出这个单词个各个形式与时态与例句
        内容短小精悍
        单词是：{user_input}
        """
        response = openai_streaming(msg,[]) 
        placeholder = st.empty()
        full_response = ''
        for item in response:
            text = item.content
            if text is not None:
                full_response += text
                placeholder.markdown(full_response)
        placeholder.markdown(full_response)
        st.session_state.study_tate[user_input]=full_response

if user_input not in '':
    st.session_state.user_input=user_input

infobox = st.empty()

key = st.sidebar.selectbox('单词列表',st.session_state.study_tate.keys())

if st.sidebar.button("显示") and key is not None:
    infobox.markdown(st.session_state.study_tate[key])

if st.sidebar.button('测试'):
    None