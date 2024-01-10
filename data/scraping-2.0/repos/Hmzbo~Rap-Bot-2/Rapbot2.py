# import libraries
import sys
import argparse

import os
import numpy as np

import openai
import requests

import gradio as gr

from TT2_FakeYou import *

try:
    import credentials
except:
    raise Exception("No 'credentials.py' file in current directory!")


# check initial project structure
if "fakeyou" not in os.listdir():
    raise Exception("Can't find the sub-directory 'fakeyou' in current directory!")
else:
    if "__init__.py" not in os.listdir('./fakeyou'):
        raise Exception("Can't find the file '__init__.py' in 'fakeyou' sub-directory!")
    

# setting openai API variables
POST_url = "https://api.openai.com/v1/chat/completions"
openai.api_key = credentials.API_KEY_OPENAI


def openai_request(history, temp):
    # create API request payload & headers
    payload = {
    "model": 'gpt-3.5-turbo',
    "messages": history,
    "temperature": temp,
    "top_p": 1.0,
    "n": 1,
    "stream": False,
    "presence_penalty": 0,
    "frequency_penalty": 0
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai.api_key}"
    }

    # send POST request
    response = requests.post(POST_url, headers=headers, json=payload, stream=False)

    return response.json()['choices'][0]["message"]

def chatbot_history(history):
    # create a list of tuples (user msg, gpt response) for the gradio chatbot component.
    chatbot_hist=[]
    for i in range(1, len(history), 2):
        chatbot_hist.append((history[i]["content"],history[i+1]["content"]))
    return chatbot_hist

def chatgpt_clone(input, history, params):
    # record conversation history with the chatbot
    if not history:
        if params['style']=="None":
            history.append({"role": "system", "content": f"You are a skilled rapper."})
        else:
            history.append({"role": "system", "content": f"You are a skilled {params['style']} rapper."})
    # define prompt structure and sent API request
    prompt = {"role":"user", "content": input}
    response = openai_request(history, params['temp'])

    # update chat history 
    history.append(prompt)
    history.append(response)
    chatbot_hist = chatbot_history(history)
    
    return chatbot_hist, history


with gr.Blocks(theme='finlaymacklon/smooth_slate') as blocks:
    gr.Markdown("""<h1><center>Rap Bot</center></h1>""")
    with gr.Row():
        with gr.Column(scale=0.8):
            gr.Markdown("""<p>This rap bot allows you to:</p>
            <ol>
            <li>Generate great rap lyrics.</li>
            <li>Clone real rapper voices singing the generated lyrics.</li>
            </ol>
            """)
        with gr.Column(scale=0.2):
            error_box = gr.Textbox(value="App Running..", show_label=False, interactive=False)

    def get_params(slider,radio,params):
        params['temp']=slider
        params['style']=radio
        return params
    

    with gr.Tab(label="Generate lyrics"):
        gr.Markdown("""This chatbot is running on ChatGPT API.\\
                        Consider yourself talking to a very skilled rapper.\\
                        Generate lyrics and copy them for audio generation.""")
        with gr.Row():
            with gr.Column(scale=0.3):
                temp_slider = gr.Slider(0,2, value=1, step=0.01, label="Creativity",
                                         info="This parameter indicates the degree of randomness for the rap bot model.")
                rap_style_radio = gr.Radio(["Old school","Trap","Clean","Mumble","None"], label="Style", value="None")
                set_params_btn = gr.Button("Set parameters")
                params = gr.State(value={'temp':1, 'style':"None"})
                set_params_btn.click(get_params, inputs=[temp_slider,rap_style_radio,params], outputs=params)
                draft_box = gr.TextArea(label='Draft', max_lines=50,
                                         placeholder="Use this area as notepad to edit or adjust generated lyrics.")
            with gr.Column(scale=0.7):
                chatbot = gr.Chatbot(label="Talk to me!")
                state = gr.State([])
                with gr.Row():
                    message = gr.Textbox(show_label=False,placeholder='What kind of rap lyrics you want to generate?')
                    send_btn = gr.Button("SEND").style(full_width=False, size='sm')
                send_btn.click(chatgpt_clone, inputs=[message, state, params], outputs=[chatbot, state])
    
    with gr.Tab(label="Generate voice"):
        gr.Markdown("""## How to use?
                        1. Set parameters for Tacotron 2 & HiFi-GAN models.
                        2. Initialize Tacotron 2 (May take few minutes when running for 1st time).
                        3. Enter lyrics.
                        4. Generate audio.""")
        with gr.Row():
            with gr.Column(scale=0.3):
                params_state = gr.State({'initialized':False})

                tacotron_id = gr.Textbox(label="Tacotron2 ID", placeholder="Enter Tacotron2 trained model name.")
                hifigan_id = gr.Textbox(label="HiFi-GAN Model", value="universal",
                                         info='Default model is "Universal" but has some robotic noise.\
                                              Provide Google Drive ID to use a custom model.')
                pronounciation_dic_box = gr.Checkbox(label="Pronounciation Dict",value=False)
                show_graphs_box = gr.Checkbox(label="Show Graphs", value=True)
                max_duration_field = gr.Number(value=20, label='Max Duration')
                stop_threshold_field = gr.Number(value=0.5, label='Stop Threshold')
                superres_strength_field = gr.Number(value=10, label='Super Resolution Strength',
                                                     info='If the audio sounds too artificial, you can lower the superres_strength')
                set_params_btn = gr.Button(value="Set Parameters")

                input_list=[params_state,error_box,tacotron_id,hifigan_id,pronounciation_dic_box,show_graphs_box,
                            max_duration_field,stop_threshold_field,superres_strength_field]
                set_params_btn.click(get_tt2_params, inputs=input_list, outputs=[params_state,error_box])
                gr.Markdown("---")
                gr.Markdown("""<span style="color:grey">Always re-initialize Tactron 2 after changing parameters.</span>""")
                gr.Markdown("---")
                initialize_btn = gr.Button(value="Initialize Tacotron2")
                initialization_status = gr.Textbox(label="Initialization status")
                initialize_btn.click(initialize_tacotron2, inputs=[params_state],
                                      outputs=[initialization_status, params_state])

            with gr.Column(scale=0.7):

                lyrics_box = gr.TextArea(label='Lyrics', placeholder='Enter lyrics here')
                generate_audio_btn = gr.Button(value='Generate Audio')
                audio_player = gr.Audio(label='Result Audio')
                with gr.Row():
                    result_image1 = gr.Image(label='Results Image 1', shape=(450,360))
                    result_image2 = gr.Image(label='Results Image 2', shape=(450,360))
                generate_audio_btn.click(end_to_end_infer, inputs=[lyrics_box, params_state],
                                          outputs=[audio_player, result_image1, result_image2])
                gr.Markdown("---")
                gr.Markdown("---")
                update_tt2_box = gr.Textbox(label='Tacotron2 ID', placeholder='Enter new TT2 model name')
                update_hifigan_box = gr.Textbox(label='HiFi-GAN Model', placeholder='Enter new HiFi-GAN model name')
                update_models_btn = gr.Button(value='Update models')
                update_models_btn.click(update_tt2_model, inputs=[params_state,update_tt2_box,update_hifigan_box],
                                         outputs=[params_state])
                
        gr.Markdown("""## Common Error messages:
                        1. 'No TACOTRON2 ID provided': The user forget to input a Tacotron 2 ID.
                        2. 'Invalid Tacotron ID': Can't find Tacotron 2 model with the provided ID in fakeyou sub-directory or in Google drive. Or the download from G-Drive failed!
                        """)
                

parser = argparse.ArgumentParser(description="Rapbot args parser",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--share", action="store_true", help="For shareable gradio app link.")
parser.add_argument("--debug", action="store_true", help="To activate gradio debugging mode.")
parser.add_argument("--show-err", action="store_true", help="To display Errors on the UI and browser console log.")
args = parser.parse_args()
config = vars(args)

blocks.queue().launch(inbrowser=True, show_error=config["show_err"], debug = config["debug"], share=config["share"])
