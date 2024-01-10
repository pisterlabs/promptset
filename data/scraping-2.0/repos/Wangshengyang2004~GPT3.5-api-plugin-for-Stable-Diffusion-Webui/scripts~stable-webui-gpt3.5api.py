# Import required modules
import os
import json
import openai
import gradio as gr
import modules
from pathlib import Path
from modules import script_callbacks
import modules.scripts as scripts
from config_private import *

openai.api_key = OPENAI_API_KEY

#os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
#os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

base_dir = scripts.basedir()

class ChatGPT:
    def __init__(self, user):
        self.user = user
        self.messages = [{"role": "system", "content": "You are an instructor of the painter"}]

    def ask_gpt(self, system_msg, user_msg):
        self.messages.append({"role": "system", "content": system_msg})
        self.messages.append({"role": "user", "content": user_msg})
        rsp = openai.ChatCompletion.create(
          model="gpt-3.5-turbo-0613",
          messages=self.messages
        )
        self.messages.append({"role": "assistant", "content": rsp.get("choices")[0]["message"]["content"]})
        return rsp.get("choices")[0]["message"]["content"]

    def clear_history(self):
        self.messages = []
        return "History cleared."

def add_to_prompt(prompt):  # A holder TODO figure out how to get rid of it
    return prompt

def on_ui_tabs():
    # Instantiate ChatGPT with user "user"
    chat_gpt = ChatGPT("user")
    txt2img_prompt = modules.ui.txt2img_paste_fields[0][0]
    img2img_prompt = modules.ui.img2img_paste_fields[0][0]

    with gr.Blocks(analytics_enabled=False) as prompt_generator:
        with gr.Column():
            with gr.Row():
                system_textbox = gr.Textbox(lines=2, placeholder="System Message", label="System Message")
                user_textbox = gr.Textbox(lines=2, placeholder="User Message", label="User Message")
        with gr.Column():
            with gr.Row():
                generate_button = gr.Button(value="Generate", elem_id="generate_button")
                clear_button = gr.Button(value="Clear", elem_id="clear_button")
        results_vis = []
        results_txt_list = []
        with gr.Tab("Results"):
            with gr.Column():
                result_textbox = gr.Textbox(label="", lines=3)
                
            with gr.Column(scale=1):
                txt2img = gr.Button("send to txt2img")
                img2img = gr.Button("send to img2img")
                
            # Handles ___2img buttons
            txt2img.click(add_to_prompt, inputs=[
                result_textbox], outputs=[txt2img_prompt]).then(None, _js='switch_to_txt2img',
                                                                                inputs=None, outputs=None)
            img2img.click(add_to_prompt, inputs=[
                result_textbox], outputs=[img2img_prompt]).then(None, _js='switch_to_img2img',
                                                                                inputs=None, outputs=None)
            results_txt_list.append(result_textbox)
            generate_button.click(fn=chat_gpt.ask_gpt, inputs=[system_textbox, user_textbox], outputs=result_textbox)
            clear_button.click(fn=chat_gpt.clear_history, inputs=[], outputs=result_textbox)

 
    return (prompt_generator, "GPT3.5 API Prompt Generator", "GPT3.5 API Prompt Generator"),

# Run the on_ui_tabs function when the UI tabs are activated
script_callbacks.on_ui_tabs(on_ui_tabs)
