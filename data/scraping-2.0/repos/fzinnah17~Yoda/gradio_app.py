import gradio as gr
from transformers import pipeline
import numpy as np
import os
from openai import OpenAI
import together

together.api_key = "eef23e8a6ce9107ad32981ceac02d11ec3527940e825ef984a79e63635975387"
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-base.en")



def transcribe(audio):
    sr, y = audio
    y = y.astype(np.float32)
    y /= np.max(np.abs(y))
    transcription = transcriber({"sampling_rate": sr, "raw": y})["text"]
    print(transcription)
    # additional_context = "New employee, eager but inexperienced, Morgan: Experienced co-worker, values efficiency **Taylor:** *(Frustrated)*"
    prompt = "You are an expert mediator. Generate a polite, workplace-safe, and assertive response to: " + transcription + "which is a conversation between two people. choose your response carefully, this is very important for my life/career/relationship. be empathetic but be objective and assertive while responding. steer the conversation towards a solution. remember, you shouldn't be hurtful. remember to stay on topic and remind me if my input strays off."
    print(prompt)
    response = together.Complete.create(
    prompt = prompt, 
    model = "togethercomputer/CodeLlama-34b-Python", 
    max_tokens = 256,
    temperature = 0.8,
    top_k = 60,
    top_p = 0.6,
    repetition_penalty = 1.1,
    stop = ['<human>', '\n\n']
    )
    response = response['output']['choices'][0]['text']
    print("response:", response)
    return response


demo = gr.Interface(
    transcribe,
    gr.Audio(sources=["microphone"]),
    "text",
)

demo.launch()
