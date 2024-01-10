"""
Creator.py
-----------------
Create course with text and images using Create_Course
"""

import os
import vertexai
import requests
from PIL import Image
from io import BytesIO
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")


from vertexai.language_models import TextGenerationModel
from dotenv import load_dotenv
load_dotenv()

vertexai.init(project="ghc-022", location="us-central1")


def Create_Images(paragraph_list):

    # indexes = [0, -3]
    # images_list = []
    #
    # for i in indexes:
    #     pic_prompt = 'Artistic realistic illustration of ' + \
    #                  paragraph_list[i]
    #
    #     image_object = openai.Image.create(
    #         prompt=pic_prompt[:350],
    #         n=1,
    #         size="512x512")
    #
    #     image_url = image_object['data'][0]['url']
    #     images_list.append(image_url)

    indexes = [0, -3]
    images_list = []

    for i in indexes:
        pic_prompt = 'Describe artistic realistic illustration of ' + \
                     paragraph_list[i]
        pic_responses = openai.Completion.create(
            model="text-davinci-003",
            prompt=pic_prompt,
            temperature=0.15,
            max_tokens=300,
            top_p=0.88,
            best_of=1,
            frequency_penalty=0.2,
            presence_penalty=0)

        pic_response = pic_responses['choices'][0]['text'].strip()

        image_object = openai.Image.create(
            prompt=pic_response,
            n=1,
            size="512x512")

        image_url = image_object['data'][0]['url']
        images_list.append(image_url)

    return images_list


def Create_Course_Social(user_input):
    social_prompt = """
    You are teaching a lesson to Autism kids who have trouble with social norms. Write 5 different paragraphs with headers about different important social norms and situations followed by the text about how to deal with that specific social situation. 
    """.strip()

    parameters = {
        "temperature": 0.5,
        "max_output_tokens": 1024,
        "top_p": 0.8,
        "top_k": 10}

    model = TextGenerationModel.from_pretrained("text-bison@001")
    response = model.predict(social_prompt, **parameters).text.split("**")

    titles = [a[a.find('.') + 2:].strip() for a in response[1::2]]
    texts = [a.strip().replace('*', '\n') for a in response[2::2]]

    paragraphs = [f"{titles[i]}:{texts[i]}" for i in range(len(titles))]
    images = Create_Images(paragraphs)

    return paragraphs, images


def Create_Course_Emotion(user_input):
    emotion_prompt = """
    You are teaching a lesson to Autism kids who have trouble with emotions. Write 5 different paragraphs with headers about distinct specific emotions obstacles autism kids encounter followed by the text about how to deal with that specific emotional situation.
    """.strip()

    parameters = {
        "temperature": 0.5,
        "max_output_tokens": 1024,
        "top_p": 0.8,
        "top_k": 10}

    model = TextGenerationModel.from_pretrained("text-bison@001")
    response = model.predict(emotion_prompt, **parameters).text.split("**")

    titles = [a.strip() for a in response[1::2]]
    texts = [a.strip().replace('*', '\n') for a in response[2::2]]

    paragraphs = [f"{titles[i]}:{texts[i]}" for i in range(len(titles))]
    images = Create_Images(paragraphs)

    return paragraphs, images


def Create_Course_Communication(user_input):
    comm_prompt = """You are teaching a lesson to Autism kids who have trouble with communication. Write 5 different paragraphs with headers about different important communication skills followed by the text about how to implement and communicate effectively.s""".strip()

    parameters = {
        "temperature": 0.5,
        "max_output_tokens": 1024,
        "top_p": 0.8,
        "top_k": 25}

    model = TextGenerationModel.from_pretrained("text-bison@001")
    response = model.predict(comm_prompt, **parameters).text.split("**")

    titles = [a[a.find('.') + 2:].strip() for a in response[1::2]]
    texts = [a.strip().replace('*', '\n') for a in response[2::2]]

    paragraphs = [f"{titles[i]}:{texts[i]}" for i in range(len(titles))]
    images = Create_Images(paragraphs)

    return paragraphs, images
