
import os
from dotenv import find_dotenv,load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI
import requests
import streamlit as st

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN=os.getenv("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")

#img2text
def img2text(url):
    image_to_text= pipeline('image-to-text', model='Salesforce/blip-image-captioning-base')
    
    text = image_to_text(url)[0]['generated_text']
    print(text)
    return text

#img2text("image1.jpg")

## LLMs
def generate_story(scenario):
    template="""
    You are a story tellar;
    You can generate a short story based on a simple narrative, the story should be no more than 20 words;
    
    CONTEXT:{scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=['scenario'])
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=1)
    
    story_llm = LLMChain(llm=llm, prompt=prompt, verbose=True)
    
    story = story_llm.predict(scenario=scenario)
    print(story)
    return story


# text to speech
def text2speech(message):

    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": "Bearer {HUGGINGFACEHUB_API_TOKEN}"}

    payloads = {
        "inputs":message
    }
        
    response = requests.post(API_URL, headers=headers, json=payloads)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)

scenario = img2text("image1.jpg")
story = generate_story(scenario)
text2speech(story)

## if __name__ == '__main__':
    