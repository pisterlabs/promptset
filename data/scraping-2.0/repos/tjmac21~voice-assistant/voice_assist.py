import gradio as gr
from openai import OpenAI
import os

client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'] 
)
messages = [{"role": "system", "content": 'You are an expert software engineer helping me interview. Your task is to help me answer coding and algorithm questions. Be as straight as terse as possible. Only add necessary punctuation such as periods, commas, and capitalization, and use only the context provided.'}]

def transcribe(audio):
    global messages

    audio_filename_with_extension = audio + '.wav'
    os.rename(audio, audio_filename_with_extension)
    
    audio_file = open(audio_filename_with_extension, "rb")
    transcript = client.audio.transcriptions.create(model="whisper-1", file=audio_file)

    messages.append({"role": "user", "content": transcript.text})
    print(f"{messages[-1]['role']}: {messages[-1]['content']}")

    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages)

    system_message = response.choices[0].message
    messages.append(vars(system_message))

    print(f"{messages[-1]['role']}: {messages[-1]['content']}")

    chat_transcript = ""
    for message in messages:
        if message['role'] != 'system':
            chat_transcript += message['role'] + ": " + message['content'] + "\n\n"

    return chat_transcript

ui = gr.Interface(fn=transcribe, inputs=gr.Audio(type="filepath"), outputs="text").launch()
ui.launch()
