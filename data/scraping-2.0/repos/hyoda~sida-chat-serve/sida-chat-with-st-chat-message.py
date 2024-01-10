import os
import random
import openai
import streamlit as st
import numpy as np
from streamlit_chat import message
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Streamlit에서 환경 변수 설정을 통해 tokenizers의 병렬 처리를 비활성화합니다.
os.environ["TOKENIZERS_PARALLELISM"] = "false"

DB_DIR = "./db/"

db = Chroma(
    embedding_function=HuggingFaceEmbeddings(model_name="jhgan/ko-sbert-sts"),
    persist_directory=DB_DIR,
)

openai.api_key = st.secrets["pass"]

# st.title("시다 챗: 남해워케이션에서 만든 인공지능 챗봇")
st.markdown(
    """<h1>시다봇 (sidabot) <small style="font-size:18pt">남해워케이션 인공지능 챗봇</small></h1> """,
    unsafe_allow_html=True,
)

st.session_state["openai_model"] = st.radio(
    "모델 선택", ("gpt-3.5-turbo", "gpt-4"), disabled=True
)
st.session_state["openai_model"] = "gpt-3.5-turbo"

search_k, pick_k = 25, 9
if st.session_state["openai_model"] == "gpt-4":
    search_k, pick_k = 50, 15

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    doc = db.similarity_search(prompt, k=search_k)
    doc = [x.page_content for x in doc if len(x.page_content) > 80]
    ref_content = "\n\n".join(random.choices(doc, k=min(pick_k, len(doc))))

    ref_prompt = f"""<reference>{ref_content}</reference>
---
질문:
{prompt}"""
    print(ref_prompt)  # prompt 테스트 출력

    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        full_response = ""
        for response in openai.ChatCompletion.create(
            model=st.session_state["openai_model"],
            messages=[
                {
                    "role": "system",
                    "content": """Serves as a truncated guide to information about Namhae-gun, Gyeongsangnam-do, South Korea. 
if the question doesn't need to refer to reference, answer it without referring to reference and ignore it.  
else if the question requires a reference, answer with a reference. 
else if the reference doesn't have enough information to answer the question, answer with "답변을 위한 충분한 자료가 확보되지 않았습니다".
한글로 답변합니다.""",
                }
            ]
            + [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[-6:-1]
            ]
            + [{"role": "user", "content": ref_prompt}],
            stream=True,
        ):
            full_response += response.choices[0].delta.get("content", "")
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
        print(full_response)  # prompt 테스트 출력
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# st.session_state["openai_model"] = st.radio(
#     "모델 선택",
#     ("gpt-3.5-turbo", "gpt-4"))

# search_k, pick_k = 25, 9
# if st.session_state["openai_model"] == 'gpt-4':
#     search_k, pick_k = 50, 15

# if "messages" not in st.session_state:
#     st.session_state.messages = []

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# if prompt := st.chat_input("What is up?"):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)

#     doc = db.similarity_search(prompt, k=search_k)
#     doc = [x.page_content for x in doc if len(x.page_content) > 80]
#     ref_content = "\n\n".join(random.choices(doc, k=min(pick_k, len(doc))))

#     ref_prompt = f"""
#     <reference>{ref_content}</reference>\n
#     reference를 참고해서 답변합니다. 최종 정답만 말합니다.
#     ---
#     {prompt}"""
#     print(ref_prompt) # prompt 테스트 출력

#     with st.chat_message("assistant"):
#         message_placeholder = st.empty()

#         full_response = ""
#         for response in openai.ChatCompletion.create(
#             model=st.session_state["openai_model"],
#             messages=[
#                 {"role": m["role"], "content": m["content"]}
#                 for m in st.session_state.messages[:-1]
#             ] + [{"role": "user", "content": ref_prompt}],
#             stream=True,
#         ):
#             full_response += response.choices[0].delta.get("content", "")
#             message_placeholder.markdown(full_response + "▌")
#         message_placeholder.markdown(full_response)
#     st.session_state.messages.append({"role": "assistant", "content": full_response})
