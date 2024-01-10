from dotenv import load_dotenv
import os
from openai import OpenAI
import streamlit as st
import time

load_dotenv()
API_KEY = os.environ['OPENAI_API_KEY']
NOVEL_AUTHOR_ASST = os.environ['NOVEL_AUTHOR_ASST']
NOVEL_AUTHOR_THREAD_1 = os.environ['NOVEL_AUTHOR_THREAD_1']

client = OpenAI(api_key=API_KEY)

# 스레드 ID 하나로 관리하기
if 'thread_id' not in st.session_state:
    thread = client.beta.threads.create()
    st.session_state.thread_id = thread.id

assistant_id = NOVEL_AUTHOR_ASST
# thread_id = NOVEL_AUTHOR_THREAD_1
thread_id = st.session_state.thread_id

# 메세지 모두 역순으로 불러오기
thread_messages = client.beta.threads.messages.list(thread_id, order="asc")

st.header("현진건 작가님과의 대화")

# 가져온 메세지 내용 UI에 표시
for msg in thread_messages.data:
    with st.chat_message(msg.role):
        st.write(msg.content[0].text.value)

# 입력창에 입력을 받아 입력된 내용으로 메세지 생성
prompt = st.chat_input("물어보고 싶은 것을 입력하세요!")

if prompt:
    # 입력 내용에 대한 메세지 생성 후 저장
    message = client.beta.threads.messages.create(
        thread_id = thread_id,
        role = "user",
        content = prompt
    )

    # 생성된 메세지 UI에 표시
    with st.chat_message(message.role):
        st.write(message.content[0].text.value)

    # RUN 작동하기
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=assistant_id,
    )

    with st.spinner('응답 기다리는 중...'):
        # RUN completed 되었나 1초마다 체크
        while run.status != "completed":
            print("status 확인 중", run.status)
            time.sleep(1)
            run = client.beta.threads.runs.retrieve(
                thread_id=thread_id,
                run_id=run.id
            )
    
    # RUN completed 되어 메세지 불러오기
    messages = client.beta.threads.messages.list(
        thread_id=thread_id
    )

    # 마지막 메세지 UI 불러오기
    with st.chat_message(messages.data[0].role):
        st.write(messages.data[0].content[0].text.value)
