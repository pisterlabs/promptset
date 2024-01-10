# Adding the imports
import openai
import deepgram
import os
import json
import requests

# Loading the API keys from the environment variables
from dotenv import load_dotenv
load_dotenv()
openai.organization = os.getenv('OPENAI_ORGANIZATION')
openai.api_key = os.getenv("OPENAI_API_KEY")
deepgram_access_code = os.getenv("DEEPGRAM_ACCESS_CODE")

# Creating the function to generate images from text using OpenAI's DALL-E model
def prompt_image_generator(prompt):
    response = openai.Image.create(
      prompt="colorful digital art in a dark synthwave style sketch "+prompt,
      n=1,
      size="256x256"
    )
    image_url = response['data'][0]['url']
    return image_url

# Creating the function to generate summary on audio using Deepgram's API
def audio_summarizer(url):
    params = {'summarize':'v2'}
    payload = {"url": url}
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "Authorization": f"Token {str(deepgram_access_code)}"
    }

    response = requests.post("https://api.beta.deepgram.com/v1/listen", json=payload, headers=headers, params= params)
    print(response.text)
    summary = response.json()['results']['summary']['short']
    return summary

def visual_keyword_extractor(summary):
    response = openai.Completion.create(
          model="text-davinci-003",
          prompt=f"""{summary}
        Give the two most important keywords, seperated by and, that can help visually describe this scene."""
        )
    return response['choices'][0]['text'].strip().replace('\n',' ')

# Creating the function to connect to the Deepgram API to DALL E
def dall_e_api(url):
    summary = audio_summarizer(url)
    visual_keywords = visual_keyword_extractor(summary)
    image_link = prompt_image_generator(visual_keywords)
    return image_link

# print(dall_e_api("https://www.wavsource.com/snds_2020-10-01_3728627494378403/people/famous/eastwood_lawyers.wav"))
print(dall_e_api("https://res.cloudinary.com/deepgram/video/upload/v1680127025/dg-audio/nasa-spacewalk-interview_ljjahn.wav"))

