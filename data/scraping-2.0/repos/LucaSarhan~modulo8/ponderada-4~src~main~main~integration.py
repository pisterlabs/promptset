#! /bin/env python3
import gradio as gr
from langchain.llms import Ollama

conversation_history = ["Answer prompts like you are a safety expert for industrial environments"]

def generate_response(prompt):
    conversation_history.append(prompt)
    full_prompt = "\n".join(conversation_history)
    opa = Ollama(base_url='http://localhost:11434', model="llama2")
    return opa(full_prompt)

def main():
    iface = gr.Interface(
        fn=generate_response,
        inputs=gr.Textbox(lines=2, placeholder="Enter your prompt here..."),
        outputs="text"
    )
    iface.launch()
    
if __name__ == "__main__":
    main()