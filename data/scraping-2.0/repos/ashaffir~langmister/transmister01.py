import os
import gradio as gr
import openai, subprocess

from dotenv import load_dotenv
load_dotenv()

messages = [{"role": "system", "content": 'You are a transcriber for Hebrew.'}]

def transcribe(audio):
    global messages

    openai.api_key = os.getenv("OPENAI_KEY")

    audio_filename_with_extension = audio + '.wav'
    os.rename(audio, audio_filename_with_extension)
    
    audio_file = open(audio_filename_with_extension, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    messages.append({"role": "user", "content": transcript["text"]})

    # response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    # system_message = response["choices"][0]["message"]
    # messages.append(system_message)

    chat_transcript = ""
    for message in messages:
        if message['role'] != 'system':
            chat_transcript += message['role'] + ": " + message['content'] + "\n\n"

    return chat_transcript

ui = gr.Interface(fn=transcribe, inputs=gr.Audio(source="microphone", type="filepath"), outputs="text").launch()
ui.launch(debug=True, share=True)
