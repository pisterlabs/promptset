import os
import time

import gradio as gr
import openai
from dotenv import load_dotenv

load_dotenv()

# 발급받은 API 키 설정
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# openai API 키 인증
openai.api_key = OPENAI_API_KEY

# 모델 - GPT 3.5 Turbo 선택
model = "gpt-3.5-turbo"

with gr.Blocks() as app:
    chatbot = gr.Chatbot()
    msg = gr.Textbox()
    clear = gr.ClearButton([msg, chatbot])

    def respond(message, chat_history):
        # 메시지 설정하기
        input_msg = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": message},
        ]

        # ChatGPT API 호출하기
        response = openai.ChatCompletion.create(model=model, messages=input_msg)
        answer = response["choices"][0]["message"]["content"]

        chat_history.append((message, answer))
        time.sleep(2)
        return "", chat_history

    msg.submit(respond, [msg, chatbot], [msg, chatbot])

if __name__ == "__main__":
    app.launch()
