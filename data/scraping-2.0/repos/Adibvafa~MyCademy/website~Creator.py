"""
Creator.py
-----------------
Create course with text and images using Create_Course
"""


import os
import openai

from PIL import Image
import requests
from io import BytesIO

import asyncio
import aiohttp

secret_api_key = ''
openai.api_key = secret_api_key


def Create_Course(user_input):
    """
    Returns list of paragraphs, and list of images for index 0 and -3
    :param user_input: Input of user
    """
    # person_prompt = "Who is a good person to create a course on the topic " + user_input + ". Give the answer as
    # maximum 4 words"

    # person_response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[{"role": "user", "content": person_prompt}],
    #     temperature=0.1,
    #     max_tokens=6,
    #     top_p=0.95,
    # )
    # person = person_response["choices"][0]["message"]["content"]

    # topics_prompt = f'Create a list of creative seven one sentence titles of topics that must be discussed in a webpage answering {user_input}. Separate each title with @@. The first topic must be introduction and topic seven must be conclusion.'

    # topics_response = openai.ChatCompletion.create(
    #     model="gpt-3.5-turbo",
    #     messages=[{"role": "user", "content": topics_prompt}],
    #     temperature=0.1,
    #     max_tokens=200,
    #     top_p=0.95)

    # topics = topics_response['choices'][0]['message']['content'].strip()
    # topics_list = topics.split('@@')

    # paragraph_list = []
    # for topic in topics_list:
    #     paragraph_list.append(Create_Paragraph(user_input, topic))

    # num_paragraph = len(paragraph_list)

    # # for i in range(num_paragraph):
    # #     paragraph_list[i] = paragraph_list[i].strip().replace('\n', ' ')

    # if len(paragraph_list[0]) < 5:
    #     paragraph_list = paragraph_list[1:]
    #     num_paragraph -= 1

    topics_list = Create_topics(user_input)
    paragraph_list = asyncio.run(
        create_course_text_async(user_input, topics_list))
    # num_paragraph = len(paragraph_list)

    # for i in indexes:
    #     pic_prompt = 'Describe artistic realistic illustration of ' + \
    #                  paragraph_list[i]
    #     pic_responses = openai.Completion.create(
    #         model="text-davinci-003",
    #         prompt=pic_prompt,
    #         temperature=0.15,
    #         max_tokens=200,
    #         top_p=0.88,
    #         best_of=1,
    #         frequency_penalty=0.2,
    #         presence_penalty=0)

    #     pic_response = pic_responses['choices'][0]['text'].strip()

    #     image_object = openai.Image.create(
    #         prompt=pic_response,
    #         n=1,
    #         size="512x512")

    #     image_url = image_object['data'][0]['url']

    #     #         url_response = requests.get(image_url)
    #     #         image = Image.open(BytesIO(url_response.content))
    #     images_list.append(image_url)

    image_prompts = asyncio.run(create_image_text_async(paragraph_list))
    image_urls = asyncio.run(course_img_async(image_prompts))

    print(paragraph_list)

    return topics_list, paragraph_list, image_urls


async def create_image_text(text):
    pic_prompt = 'Describe artistic realistic illustration of ' + text
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: openai.Completion.create(
            model="text-davinci-003",
            prompt=pic_prompt,
            temperature=0.15,
            max_tokens=200,
            top_p=0.88,
            best_of=1,
            frequency_penalty=0.2,
            presence_penalty=0)
    )
    one_img_descrip = response['choices'][0]['text'].strip()
    return one_img_descrip


async def create_image_text_async(paragraph_list):
    image_text_list = await asyncio.gather(*(create_image_text(text) for text in paragraph_list))
    return image_text_list


async def create_image(prompt):
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: openai.Image.create(
            prompt=prompt,
            n=1,
            size="256x256"
        )
    )
    image_url = response["data"][0]["url"]
    return image_url


async def process_image(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            image_data = await response.read()


async def course_img_async(image_text_list):
    image_urls = await asyncio.gather(*(create_image(p) for p in image_text_list))
    await asyncio.gather(*(process_image(url) for url in image_urls))
    return image_urls


def Create_Paragraph(user_input, topic):
    """
    Crates one paragraph based on the given topic title
    :param user_input:
    :param topic:
    :return:
    """
    paragraph_prompt = f'You are creating a webpage article based on {user_input} and now you are creating the body of the paragraph about {topic}' + \
        'Explain with accurate detail and use engaging clear understandable sentences and do not include any title in your answer.'

    paragraph_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": paragraph_prompt}],
        temperature=0.1,
        max_tokens=200,
        top_p=0.95)

    return paragraph_response['choices'][0]['message']['content'].strip()


def Create_topics(user_input):
    """
    Returns list of paragraphs, and list of images for index 0 and -3
    :param user_input: Input of user
    """
    topics_prompt = f'Create a list of numbered creative seven one sentence titles of topics that must be discussed in a webpage answering {user_input}. Separate each title with newline. The first topic must be introduction and topic seven must be conclusion.'
    topics_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": topics_prompt}],
        temperature=0.1,
        max_tokens=200,
        top_p=0.95)

    topics = topics_response['choices'][0]['message']['content'].strip()
    topics_list = topics.split('\n')
    return topics_list


async def create_text(user_input, topic):
    loop = asyncio.get_event_loop()
    main_prompt = f'You are creating a webpage based on {user_input}\
        and now you are creating the paragraph about {topic}' + 'Explain with accurate detail and \
        use engaging clear understandable sentences and do not include any title in your answer.'
    response = await loop.run_in_executor(
        None,
        lambda: openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": main_prompt}],
            temperature=0.1,
            max_tokens=250,
            top_p=0.95,
        )
    )
    one_parah = response["choices"][0]["message"]["content"].strip()
    return one_parah


async def create_course_text_async(user_input, topic_list):
    paragraph_list = await asyncio.gather(*(create_text(user_input, t) for t in topic_list))
    return paragraph_list
    # paragraph_list = await asyncio.gather(*(create_text(t) for t in topic_list))
    # await asyncio.gather(*(process_text(txt) for txt in paragraph_list))
    # return paragraph_list
