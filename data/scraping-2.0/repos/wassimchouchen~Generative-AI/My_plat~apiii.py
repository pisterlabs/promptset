import subprocess
import requests
import torch
import json
import openai
from PIL import Image  # Import the required libraries
import base64
from io import BytesIO
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import whisper
from diffusers import StableDiffusionPipeline
# My OpenAI API key
openai.api_key = "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"
# My hugging_face API key
headers = {"Authorization": "Bearer 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"}

def ASR_WHISPER(payload) :
        file = open(payload, "rb")
        response = openai.Audio.transcribe("whisper-1", file)
        return (response["text"]) 

# Use the text generation function to generate a response
def GPT35(payload) :
        response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", 
        messages=[{"role": "user", "content":payload }])
        return (response["choices"][0]["message"]["content"]) 



API_URL_fr = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-fr"
def french_translator(payload):
	response = requests.post(API_URL_fr, headers=headers, json=payload)
	return response.json()
	

 


API_URL_translator = "https://api-inference.huggingface.co/models/t5-small"
def Translator(payload):
	response = requests.post(API_URL_translator, headers=headers, json=payload)
	return response.json()



API_URL_en2ar  = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-en-ar"
def en2ar(payload):
	response = requests.post(API_URL_en2ar, headers=headers, json=payload)
	return response.json()
  


API_URL_ar2en = "https://api-inference.huggingface.co/models/Helsinki-NLP/opus-mt-ar-en"
def ar2en(payload):
	response = requests.post(API_URL_ar2en, headers=headers, json=payload)
	return response.json()


API_URL_Summarizer = "https://api-inference.huggingface.co/models/sshleifer/distilbart-xsum-12-3"
def Summarizer(payload):
    data = json.dumps(payload)
    response = requests.request("POST", API_URL_Summarizer, headers=headers, data=data)
    return json.loads(response.content.decode("utf-8"))

API_URL_QA = "https://api-inference.huggingface.co/models/deepset/roberta-base-squad2"
def QA(payload):
	response = requests.post(API_URL_QA, headers=headers, json=payload)
	return response.json()