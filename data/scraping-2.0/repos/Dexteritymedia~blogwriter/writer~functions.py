import os
import urllib.request
import re
from time import time, sleep

from django.conf import settings

import openai

openai.api_key = settings.OPENAI_API_KEY

def generateyoutubevideo(search):
    name = search
    if name:
        query = name.replace(' ', '+')
        html = urllib.request.urlopen("https://www.youtube.com/results?search_query="+query)
        video_ids = re.findall(r"watch\?v=(\S{11})", html.read().decode())
        return video_ids[:3]
    else:
        return None

def generateblogoutline(audience, title):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Generate 5-6 detailed blog outlines for {} based on this\nTitle: {}".format(audience, title),
        temperature=0.7,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)

    return response['choices'][0]['text'].replace('\n','<br/>')

def generateblogpost(outline):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="""
                Write a 2500 words blog post explaining each outlines, headings
                and subtitles in an informative and expertise format. Your writing
                style should demonstrate a deep understanding of the topic and a shows
                you have experience. \n\nMake sure to include semantic adjectives and
                do not forget to add appropriate html tags and also add a clickable
                table of contents with the heading tag h2 and ul tag for the lists.
                \n\nMake sure the first letter in headings start with capital letter
                \n\n{}""".format(outline),
        temperature=0.7,
        max_tokens=2805,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)

    return response['choices'][0]['text']

def regenerateblogpost(blog_post):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="""
                Rewrite and paraphase the article below, add appropriate html tags
                and a clickable table of contents and showcase you are an expert using
                Dave Ramsey style of writing\n\nArticle:{}\n\nResult:""".format(blog_post),
        temperature=0.85,
        max_tokens=2000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)

    return response['choices'][0]['text']


def generatemetadescription(title):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Create an SEO meta description for this blog post {}".format(title),
        temperature=0.7,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)

    return response['choices'][0]['text']


def generateyoutubelink(topic):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="""
                Generate 3 active youtube links in html tags with
                complete url on the following topic separate each video
                link with the tag <br/>\nTopic: {}""".format(topic),
        temperature=0.7,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)

    return response['choices'][0]['text']


def addyoutubelink(link, post):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Add this youtube link {} to the appropriate section to this blog post:\n {}".format(link, post),
        temperature=0.3,
        max_tokens=2500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)

    return response['choices'][0]['text']


def gpt_completion(post, model="text-davinci-003", tokens=2000, temp=0.7, top_p=1, freq_pen=0, pres_pen=0 ):
    
    max_retry = 5
    retry = 0
    while True:
        try:
            response = openai.Completion.create(
                model=model,
		prompt="Rewrite this article and make sure to keep the appropriate html tags.\nArite:\n\n{}\n\n".format(post),
		temperature=temp,
		max_tokens=tokens,
		top_p=top_p,
		frequency_penalty=freq_pen,
		presence_penalty=pres_pen
                )
            text = response['choices'][0]['text'].strip()
            return text
        except Exception as e:
            retry += 1
            if retry >= max_retry:
                return "GPT3 error: %s" % e
            print('Error communication with OpenAI:', e)
            sleep(1)
