"""
Creator.py
-----------------
Create course with text and images using Create_Course
"""

import os
import openai

openai.api_key = "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"

from PIL import Image
import requests
from io import BytesIO


def Create_Course(user_input):
    """
    Returns list of paragraphs, and list of images for index 0 and -3
    :param user_input: Input of user
    """
    person_prompt = "Who is a good person to create a course on the topic " + user_input + \
                    ". Give the answer as maximum 4 words"

    person_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": person_prompt}],
        temperature=0.1,
        max_tokens=6,
        top_p=0.95,
    )
    person = person_response["choices"][0]["message"]["content"]

    begin_prompt = "You have to act as a " + person + 'Give a professional course on '
    end_prompt = """. Explain with numerous accurate detail and use engaging clear understandable sentences.
    Start with introduction, divide it to several long paragraphs and end with summarizing conclusion.
    Put @@ in the beginning of each paragraph. """
    text_prompt = begin_prompt + user_input + end_prompt

    description_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": text_prompt}],
        temperature=0.1,
        max_tokens=2000,
        top_p=0.95,
    )

    description = description_response['choices'][0]['message']['content'].strip()
    paragraph_list = description.split('@@')
    num_paragraph = len(paragraph_list)
    for i in range(num_paragraph):
        paragraph_list[i] = paragraph_list[i].strip().replace('\n', ' ')

    if len(paragraph_list[0]) < 5:
        paragraph_list = paragraph_list[1:]
        num_paragraph -= 1

    indexes = [0, -3]
    images_list = []

    for i in indexes:
        pic_prompt = 'Describe artistic realistic illustration of ' + paragraph_list[i]
        pic_responses = openai.Completion.create(
            model="text-davinci-003",
            prompt=pic_prompt,
            temperature=0.15,
            max_tokens=300,
            top_p=0.88,
            best_of=1,
            frequency_penalty=0.2,
            presence_penalty=0
        )

        pic_response = pic_responses['choices'][0]['text'].strip()

        image_object = openai.Image.create(
            prompt=pic_response,
            n=1,
            size="512x512"
        )
        image_url = image_object['data'][0]['url']

#         url_response = requests.get(image_url)
#         image = Image.open(BytesIO(url_response.content))
        images_list.append(image_url)

    return paragraph_list, images_list