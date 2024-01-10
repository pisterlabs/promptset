#! /bin/env python3
import gradio as gr
from langchain.llms import Ollama
from langchain.prompts import ChatPromptTemplate
from bs4 import BeautifulSoup
import requests

conversation_history = []

model = Ollama(model="llama")

reference = "According to the main content of:\n"
question = "Answer the following question:\n"

content = ChatPromptTemplate.from_template(
"""
{reference}
{prompt}
"""
)

def link_response(link):
    response = requests.get(link)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, 'html.parser')
    main_content = soup.find('main')
    text = main_content.get_text(separator='\n', strip=True)
    return text

def llama_response(message, history, additional):

    chain = content | model

    conversation_history.append(message)
    full_prompt = "\n".join(conversation_history)
    question_prompt = question + full_prompt + "\n"
    
    reference_response = ""
    
    if additional:
        response = link_response(additional)
        reference_response = reference + response + "\n"
        print(reference_response)

    print(question_prompt)

    answer = ""
    
    for s in chain.stream({"reference": reference_response, "prompt": question_prompt}):
        answer = answer + s
        yield answer

def main():
    llama_face = gr.ChatInterface(
        fn=llama_response,
        additional_inputs=[gr.Textbox(lines=1, label="Link for more context")],
    )
    
    llama_face.launch()

if __name__ == "__main__":
    main()