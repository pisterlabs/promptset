import openai
import config
import PyPDF2 
import re
import tiktoken
from pptx import Presentation
import openai
import config
from PIL import Image, ImageEnhance
import requests
from io import BytesIO

openai.api_key = config.api_key
encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo") 

openai.api_key = config.api_key
# //give me 10 headers for 10 slides for presentation about photoshop short answer
def image(body):
    response = openai.Image.create(
        prompt=body,
        n=1,
        size="960x540" #1024/1024
    )
    image_url = response['data'][0]['url']
    return image_url

def question(body):
    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[
            {'role': 'user', 'content': body}
        ],
        temperature=0
    )
    return response['choices'][0]['message']['content']
# print(question("give me 10 headers for 10 slides for presentation about Creative hobbies short answer").replace('\n', '').split('.')[1:])

def initial(request="Creative hobbies", number=10):
    return(question(f"give me {number} headers for {number} slides for presentation about '{request}' short answer").replace('\n', '').split('.')[1:])


def generateImages(topics):
    for i in range(len(topics)):
        url = image(str(topics[i]) + "Darkened Image")
        print(url, topics[i])


generateImages(initial())