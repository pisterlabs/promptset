import os
from django.conf import settings
import openai

openai.api_key = settings.OPENAI_API_KEY

def generate_ad_copy(text):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt="Write a persuasive social media copy for the following product\n\nProduct: {}".format(text),
        temperature=0.7,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)

    return response['choices'][0]['text'].strip()
