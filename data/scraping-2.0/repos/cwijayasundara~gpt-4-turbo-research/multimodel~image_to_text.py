import openai
import os

from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

print(openai.api_key)

from openai import OpenAI

client = OpenAI()

#  describe a picture
prompt = 'Describe the image? Who are these people? produce the output in point form'
url = 'https://awards.acm.org/binaries/content/gallery/acm/ctas/awards/turing-2018-bengio-hinton-lecun.jpg'

result = client.chat.completions.create(
    model='gpt-4-vision-preview',
    max_tokens=500,
    messages=[{
        'role': 'user',
        'content': [prompt, url]
    }]
)

print("Details about the image", end="\n")
print(result.choices[0].message.content, end="\n")

# describe a chart

prompt = 'Describe the image? produce the output in point form'
url = 'https://www.mongodb.com/docs/charts/images/charts/stacked-bar-chart-reference-small.png'

result = client.chat.completions.create(
    model='gpt-4-vision-preview',
    max_tokens=500,
    messages=[{
        'role': 'user',
        'content': [prompt, url]
    }]
)

print("Details about the chart", end="\n")
print(result.choices[0].message.content, end="\n")

# describe a local image with equations

prompt = 'Can you solve the first equation given in the image?'
url = 'https://thirdspacelearning.com/wp-content/uploads/2021/03/Solving-Equations-What-is.png'

result = client.chat.completions.create(
    model='gpt-4-vision-preview',
    max_tokens=500,
    messages=[{
        'role': 'user',
        'content': [prompt, url]
    }]
)
print("Solving Equations", end="\n")
print(result.choices[0].message.content, end="\n")

# describe a local image with equations
import base64

with open('images/img.png', 'rb') as image_file:
    img = image_file.read()

img_byte = base64.b64encode(img).decode('utf-8')

prompt = "Describe the image. produce the output in point form"

result = client.chat.completions.create(
    model='gpt-4-vision-preview',
    max_tokens=500,
    messages=[{
        'role': 'user',
        'content': [prompt, {'image': img_byte, 'resize': 768}]
    }]
)

print("Describing local images", end="\n")
print(result.choices[0].message.content, end="\n")
