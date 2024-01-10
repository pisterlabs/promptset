## This is going to be our end file for the reference.

import logging
from PyDictionary import PyDictionary
logging.basicConfig(level=logging.INFO)
import requests
import nltk
import os
import wget
import json

from google_images_search import GoogleImagesSearch
import requests
import openai


#============This returns the definition of the word from openai API===========
openai.api_key = 'Your openai KEY'

def get_def(word, context):
    model = "text-davinci-002"
    prompt = (
        f"Please provide the definition of the word '{word}' as used in the following sentence:\n\n"
        f"{context}\n\n"
        f"Definition of '{word}': "
    )

    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )

    definition = response.choices[0].text.strip()
    return definition

#=============================================================================================#




#=============================================================#
# This function gets the pictures for our reference from Google Custom Search API
#=============================================================#
def getPicture(keyword):
    gis = GoogleImagesSearch('YOUR Google API KEY', 'YOUR custom search API KEY')


    gis.search({'q': keyword})

    dir_path = './site/'

    try:

        image_url = gis.results()[0].url

        response = requests.get(image_url, headers={'User-Agent': 'Mozilla/5.0'})
        img_data = response.content
    except IndexError:
        return None

    with open(os.path.join(dir_path, 'image.png'), 'wb') as f:
        f.write(img_data)


def getPictureGuest(keyword):
    gis = GoogleImagesSearch('YOUR Google API KEY', 'YOUR custom search API KEY')


    gis.search({'q': keyword})

    dir_path = './site/'

    try:

        image_url = gis.results()[0].url

        response = requests.get(image_url, headers={'User-Agent': 'Mozilla/5.0'})
        img_data = response.content

    except IndexError:

        return None

    with open(os.path.join(dir_path, 'image2.png'), 'wb') as f:
        f.write(img_data)


