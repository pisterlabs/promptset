from dotenv import load_dotenv, find_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain, OpenAI
import requests
import os

load_dotenv(find_dotenv())
HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")

# img2text model
def img2txt(url):
    image_to_txt = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text = image_to_txt(url)[0]['generated_text']
    print(text)
    return text

#llm
def generate_story(scenario):
    template = """
    you are a storyteller.
    you can generate a short story based on a simple narrative
    the story should be no more than 50 words

    CONTEXT: {scenario}
    STORY:
    """
    prompt = PromptTemplate(template=template, input_variables=["scenario"])

    story_llm = LLMChain(llm=OpenAI(
        model_name="gpt-3.5-turbo",
        temperature=1
    ), prompt=prompt, verbose=True)

    story = story_llm.predict(scenario=scenario)
    print(story)
    return story

# text to speech
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": "Bearer " + HUGGINGFACE_API_TOKEN}
    payload = {
        "inputs": message
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    with open('audio.flac', 'wb') as audio_file:
        audio_file.write(response.content)


path = "../test_imgs/t1.jpg"
scenario = img2txt(path)
story = generate_story(scenario)
text2speech(story)

"""
Requirements:
langchain==0.0.169
transformers[torch]==4.34.1
pillow
python-dotenv==1.0.0
"""