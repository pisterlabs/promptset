import openai
import requests
import json
import os
from rest_framework.parsers import JSONParser 
from dotenv import load_dotenv

#load of env variables
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')
leapai_api_key = os.getenv('LEAPAI_API_KEY')

#headers for leapai api
HEADERS = {
    "accept": "application/json",
    "content-type": "application/json",
    "authorization": f"Bearer {leapai_api_key}"
}

#stable diffusion v1.5 model
model_id = "8b1b897c-d66d-45a6-b8d7-8e32421d02cf"

#gpt3 api for story generation
def generate_story(topic):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Generate a 4 paragraph children's story with title about {topic} that contains a moral."}
        ]
    )
    content = completion.choices[0].message.content
    content = content.encode().decode('unicode_escape')
    title = content.split('\n')[0]
    title = title.replace('Title: ', '')
    res = content[content.find('\n'):]
    res = res.lstrip()
    prompt = generate_promt_for_stablediffusion(res)
    image_url = generate_image(topic)
    output = {'title': title, 'story': res, 'image': image_url}
    return output

#gpt3 api for prompt generation to send to leapai
def generate_promt_for_stablediffusion(story):
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"Create one text to image prompts that will be suitable as the title image of the below given story. Do not include the character names, instead include only the characters physical description.\n\n{story}"}
        ]
    )
    content = completion.choices[0].message.content
    content = content.encode().decode('unicode_escape')
    if ':' in content:
        content = content[content.find(':')+1:]
    content = content.strip()
    return content

#generate image from leapai
def generate_image(prompt):
    image_url = ""
    inference_id, status = generate_instance(
        prompt=prompt,
    )
    while True:
        if inference_id is None:
            continue
        status = ""
        inference_id, status, images = get_inference_job(inference_id)
        if status == "finished":
            image_url = images[0]["uri"]
            break
    
    return image_url

#creates a instance for the leap ai to generate image
def generate_instance(prompt):
    url = f"https://api.tryleap.ai/api/v1/images/models/{model_id}/inferences"

    payload = {
        "prompt": prompt,  
        "steps": 30,
        "width": 512,
        "height": 512,
        "numberOfImages": 1,
        "promptStrength": 4,
        "upscaleBy": "x1",
        "negativePrompt": "",
        "sampler": "ddim",
        "restoreFaces": False,
    }

    response = requests.post(url, json=payload, headers=HEADERS)
    data = json.loads(response.text)

    if "error" in data:
        print("Error: ", data)
        return None, None
    inference_id = data["id"]
    status = data["state"]

    print(f"Generating Inference: {inference_id}. Status: {status}")

    return inference_id, status

#checks the status of the inference
def get_inference_job(inference_id):
    url = f"https://api.tryleap.ai/api/v1/images/models/{model_id}/inferences/{inference_id}"

    response = requests.get(url, headers=HEADERS)
    data = json.loads(response.text)

    if "id" not in data:
        print("Error: ", data)
        return None, None, None

    inference_id = data["id"]
    state = data["state"]
    images = None

    if len(data["images"]):
        images = data["images"]

    print(f"Getting Inference: {inference_id}. State: {state}")

    return inference_id, state, images

def answer_question(question, context):
    # Define the model you want to use
    model = "gpt-3.5-turbo"

    messages = [
        {"role": "system", "content": "You are an assistant that answers the questions to the children's "\
                 "story given below. You should answer the questions descriptively in a "\
                 "way that a child can understand them. If the question asked is unrelated "\
                 "to the story, do not answer the question and instead reply by asking the "\
                 "user to ask questions related to the story."},
        {"role": "user", "content": context},
        {"role": "user", "content": question}
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        max_tokens=50, 
    )

    assistant_reply = response.choices[0].message.content
    return assistant_reply
