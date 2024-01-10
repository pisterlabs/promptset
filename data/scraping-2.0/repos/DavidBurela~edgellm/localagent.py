import torch
import torchaudio
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from langchain.llms import LlamaCpp

import gradio as gr

from dotenv import load_dotenv
import os

load_dotenv()

def transcribe(file, model="openai/whisper-large"):
    print(f'Transcribing file {file}')
    generator = pipeline("automatic-speech-recognition", model=model)
    transcript = generator(file)
    print(f'Transcript: {transcript}')
    return transcript['text']

def respond(question):
    if (os.getenv('USE_OPENAI', 'false').lower() == 'true'):
        print('Using OpenAI')
        model = OpenAI()
    else:
        model = LlamaCpp(model_path=os.getenv('MODEL_PATH'), verbose=True,
                         n_threads=int(os.getenv('NUM_THREADS', '1')),
                         n_gpu_layers=int(os.getenv('NUM_GPU_LAYERS', '1')))

    template = """Question: {question}. Be **CONCISE** and answer the question with **NO ADDITIONAL DETAILS**.

    Answer: """

    prompt = PromptTemplate(template=template, input_variables=["question"])
    llm_chain = LLMChain(prompt=prompt, llm=model)

    return llm_chain.run(question)

def text_to_speech(response, lang='en', silero_voice='v3_en'):
    local_tts_model = f'silero_tts_model_{silero_voice}.pt'

    if not os.path.isfile(local_tts_model):
        torch.hub.download_url_to_file(f'https://models.silero.ai/models/tts/{lang}/{silero_voice}.pt',
                                       local_tts_model)
    device = torch.device('cpu')
    torch.set_num_threads(4)

    model = torch.package.PackageImporter(local_tts_model).load_pickle("tts_models", "model")
    model.to(device)

    sample_rate = 48000
    speaker='random'

    audio_paths = model.save_wav(text=response,
                                speaker=speaker,
                                sample_rate=sample_rate,
                                )
    print(f'Saved to {audio_paths}')
    return audio_paths

def localagent(input_file):
    to_text = transcribe(input_file)
    response_text = respond(to_text)
    response_audio = text_to_speech(response_text)
    return to_text, response_text, response_audio

with gr.Blocks(theme=gr.themes.Glass()) as local_agent:
    input = gr.Audio(source='microphone', label='question', type='filepath', format='wav')
    to_text = gr.Textbox(lines=3, label='stt')
    response_text = gr.Textbox(lines=3, label='response')
    response_audio = gr.Audio(value='test.wav')
    input.change(fn=localagent, inputs=input, outputs=[to_text, response_text, response_audio])

local_agent.launch()