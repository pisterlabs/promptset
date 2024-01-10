import gradio as gr
import openai
import random
import time

# OpenAI 라이브러리에 API 키 설정
openai.api_key = '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'

# 초기 대화 기록을 빈 리스트로 설정
initial_history = []

def chatbot_response(message, history):
    # 대화 기록을 OpenAI 형식으로 변환
    messages = [{"role": "user", "content": pair[0]} for pair in history] + [{"role": "assistant", "content": pair[1]} for pair in history]
    messages.append({"role": "user", "content": message})

    # OpenAI에 요청을 보냅니다.
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    # OpenAI로부터 받은 응답의 내용을 올바르게 추출합니다.
    bot_message = response.choices[0].message.content

    return bot_message, history

def chatbot_interface():
    chatbot = gr.Chatbot()
    msg = gr.Textbox()

    def respond(message, chat_history):
        bot_message, updated_history = chatbot_response(message, chat_history)
        updated_history.append((message, bot_message))
        return bot_message, updated_history

    with gr.Blocks():
        msg.submit(respond, [chatbot, msg], [chatbot, msg])

    return gr.Interface(fn=respond, inputs=msg, outputs=chatbot, live=True, title="ChatGPT 기반 채팅봇")

if __name__ == "__main__":
    demo = chatbot_interface()
    demo.launch()
