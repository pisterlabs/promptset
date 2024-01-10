import subprocess
import gradio as gr
import openai
import config
openai.api_key = config.OPENAI_API_KEY

messages=[
    {"role": "system", "content": "You are a helpful assistant. Respond as if you were a rapper Snoop Dogg."}
]

def speech_to_text(audio):
    '''Transcribe audio file to text by whisper.'''
    audio_file = open(audio, "rb")
    transcript = openai.Audio.transcribe('whisper-1', audio_file)
    return transcript['text']

def query_ChatGPT(messages):
    '''Send messages to OpenAI's Chat GPT-3 model and get response.'''
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
        )
    response_message = response['choices'][0]['message']['content']
    return response_message

def greet(audio):
    global messages
    transcript = speech_to_text(audio)
    messages.append({"role": "user", "content": transcript})
    response_message = query_ChatGPT(messages)
    subprocess.call(['say', response_message])
    messages.append({"role": "assistant", "content": response_message})
    output = ''
    for message in messages:
        if message['role'] != 'system':
            output += message['role'] + ': ' + message['content'] + '\n\n'
    return output

demo = gr.Interface(fn=greet, inputs=gr.Audio(source = 'microphone', type = 'filepath'), outputs="text")
demo.launch()