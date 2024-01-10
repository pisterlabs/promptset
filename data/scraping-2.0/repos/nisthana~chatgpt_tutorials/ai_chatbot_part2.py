import gradio as gr
import os
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
message_history = []
message_to_send = ""
def chat_response(message, history):
    user_message = {"role":"user","content": message}
    message_history.append(user_message)
    chat_completion = openai.ChatCompletion.create(model='gpt-3.5-turbo', messages=message_history)
    choices = chat_completion['choices']
    chatgpt_response = choices[0]['message']
    content = chatgpt_response['content']
    chatbot_response = {"role":"assistant","content": content}
    message_history.append(chatbot_response)
    return content

chat_demo = gr.ChatInterface(chat_response)
chat_demo.launch()