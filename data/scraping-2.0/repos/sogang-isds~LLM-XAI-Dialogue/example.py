import os

import gradio as gr
import openai
from langchain.chat_models import ChatOpenAI

chat_llm = ChatOpenAI(model_name="gpt-3.5-turbo")

openai.api_key = os.getenv("OPENAI_API_KEY")

from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)

messages = []


def greet(name, chat_history):
    messages.append(HumanMessage(content=name))
    ai_message = chat_llm(messages)
    messages.append(ai_message)
    response_message = ai_message.content
    chat_history.append((name, response_message))

    return "", chat_history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="채팅창")  # '채팅창'이라는 레이블을 가진 채팅봇 컴포넌트를 생성합니다.
    msg = gr.Textbox(label="입력")  # '입력'이라는 레이블을 가진 텍스트박스를 생성합니다.
    clear = gr.Button("초기화")  # '초기화'라는 레이블을 가진 버튼을 생성합니다.

    msg.submit(greet, [msg, chatbot], [msg, chatbot])  # 텍스트박스에 메시지를 입력하고 제출하면 respond 함수가 호출되도록 합니다.
    clear.click(lambda: None, None, chatbot, queue=False)  # '초기화' 버튼을 클릭하면 채팅 기록을 초기화합니다.

demo.launch()
