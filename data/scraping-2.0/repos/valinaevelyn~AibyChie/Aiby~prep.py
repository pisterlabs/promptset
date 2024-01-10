import openai
import speech_recognition as sr
import json

import keyboard

import pygame

from gtts import gTTS
import playsound
import os
import time

from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

# from langdetect import detect

# Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
# cd venv\Scripts\activate


# Set-ExecutionPolicy RemoteSigned -Scope CurrentUser


# set API key
openai.api_key = "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"

pygame.mixer.init()

memory = ConversationBufferMemory(memory_key="chat_history")
template = """You are AiBy, a virtual bank assistant, helping a human customer to solve their bank related problems. If the customer wants to do something that uses their private information like to create a new account or to transfer money, you tell them to go to the human call center. Also, if they ask about non-bank related stuff, answer you don't know.
    {chat_history}
    Human: {human_input}
    AiBy:"""
prompt = PromptTemplate(
    input_variables=["chat_history", "human_input"], template=template
) 

llm = OpenAI(temperature=0.0, openai_api_key="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")
conversation = LLMChain(
    llm=llm,
    prompt=prompt,
    # verbose=True,
    memory=memory,
    )


def load_json_file_to_AI(json_file_path):
    with open(json_file_path, 'r') as json_file:
        data = json.load(json_file)
        
        if 'questions' in data:
            for item in data['questions']:
                question = item.get('question')
                answer = item.get('answer')
                memory.save_context({"human_input": question}, {"AiBy": answer})

json_file_path = 'data\knowledge_base.json'
load_json_file_to_AI(json_file_path)

def record_audio():
    directory_path = os.path.join(os.getcwd(), 'Audio')
    filename = "input.wav"
    file_path = os.path.join(directory_path, filename)

    # print("\nRecording...\n")
    with sr.Microphone() as source:
        recognizer = sr.Recognizer()
        source.pause_threshold = 1
        audio = recognizer.listen(source, phrase_time_limit=None, timeout=None)
        with open(file_path, "wb") as f:
            f.write(audio.get_wav_data())
        
        # Transcribe audio to text
        text = transcribe_audio_to_text(file_path)
        return text

def transcribe_audio_to_text(filename):
    recognizer = sr.Recognizer()
    with sr.AudioFile(filename) as source:
        audio = recognizer.record(source)
    try:
        return recognizer.recognize_google(audio, language="id-ID")
    except:
        print("Skipping unknown error")
        return None


def speak_text(text, lang='id'):
    directory_path = os.path.join(os.getcwd(), 'Audio')
    filename = 'audioo.mp3'
    file_path = os.path.join(directory_path, filename)

    tts = gTTS(text=text, lang=lang, slow=False)
    os.remove(file_path)
    tts.save(file_path)
    time.sleep(1)
    playsound.playsound(file_path)



def get_response(text):
    response = conversation.predict(human_input=text)
    print('AI Res:' + response +'\n')
    return response
