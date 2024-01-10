import gradio as gr
from langchain.llms import OpenAI
from dotenv import load_dotenv

load_dotenv()

llm = OpenAI(model="gpt-3.5-turbo-instruct", max_tokens=512)

def openai_response(prompt, history):
    response = ""
    for chunk in llm.stream(prompt):
        response += chunk
    return response

demo = gr.ChatInterface(openai_response)

demo.launch()