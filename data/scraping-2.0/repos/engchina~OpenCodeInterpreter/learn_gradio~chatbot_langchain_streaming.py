import os

from langchain.chat_models import ChatOpenAI
# from langchain.schema import AIMessage, HumanMessage
import openai
import gradio as gr

os.environ["OPENAI_API_BASE"] = "http://192.168.31.12:8000/v1"  # Replace with your base url
os.environ["OPENAI_API_KEY"] = "sk-123456"  # Replace with your api key

llm = ChatOpenAI(temperature=1.0, model='gpt-3.5-turbo')


def predict(message, history):
    history_openai_format = []
    for human, assistant in history:
        history_openai_format.append({"role": "user", "content": human})
        history_openai_format.append({"role": "assistant", "content": assistant})
    history_openai_format.append({"role": "user", "content": message})

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=history_openai_format,
        temperature=1.0,
        stream=True
    )

    partial_message = ""
    for chunk in response:
        print(f"chunk: {chunk}")
        if len(chunk['choices'][0]['delta']) != 0:
            partial_message = partial_message + chunk['choices'][0]['delta']['content']
            yield partial_message


gr.ChatInterface(predict).queue().launch()
