## This is going to be our end file for the reference.

import logging
from PyDictionary import PyDictionary
logging.basicConfig(level=logging.INFO)

import os
from PIL import Image
from io import BytesIO

from google_images_search import GoogleImagesSearch
import requests
import openai



#============This returns the definition of the word from OpenAI===========
openai.api_key = 'Your API KEY'

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


#===================================This is Saving the Image==================================
def save_image(url, file_name, directory):
    response = requests.get(url)
    response.raise_for_status()

    img = Image.open(BytesIO(response.content))

    if not os.path.exists(directory):
        os.makedirs(directory)

    file_path = os.path.join(directory, file_name)
    img.save(file_path)

#=============================================================================================#

def bing_image_search(query, api_key, count=10):
    headers = {"Ocp-Apim-Subscription-Key": api_key}
    params = {
        "q": query,
        "count": count,
        "offset": 0,
        "mkt": "en-US",
        "safesearch": "Moderate",
    }
    url = "https://api.bing.microsoft.com/v7.0/images/search"

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()
    return response.json()



#=============================================================#
# This function gets the pictures for our reference from Google Custom Search API
#=============================================================#
def getPicture(keyword):
    gis = GoogleImagesSearch('Your API KEY', 'Your custom search KEY')

    # define the search parameters
    gis.search({'q': keyword})

    dir_path = './site/'


    # If path not availabe then do this:
    # if not os.path.exists(dir_path):
    #     os.makedirs(dir_path)

    # get the URL of the first image in the search results
    try:
        image_url = gis.results()[0].url

    # download the image from the URL and save it
    # wget.download(image_url)

        response = requests.get(image_url, headers={'User-Agent': 'Mozilla/5.0'})
        img_data = response.content
    except IndexError:
        print("No Image found")
        return None

    with open(os.path.join(dir_path, 'image.png'), 'wb') as f:
        f.write(img_data)

getPicture("a bat")




