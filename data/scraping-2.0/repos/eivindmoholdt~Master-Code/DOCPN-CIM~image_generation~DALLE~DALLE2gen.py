"""Create Image from Prompt.
this code sends an authenticated request to the API that generates a single image based on the text in PROMPT.

Script returns a JSON file with the URL for the image.
The URL is only available online for 24 hours.

The API allows you to switch the response format from a URL to the Base64-encoded image data. In line 15, you set the value of response_format to "b64_json". 

We can later Decode the Base64 string to view the image and save as a PNG

Stable Diffusion returns images in 512x512 pixel size.
Dall-E lets us choose between sizes, at different cost.
We choose 512x512 here as well to ensure comparability between the models when comparing the images at later stage.

Resolution	- Price per image
* 256×256	- $0.016

* 512×512	- $0.018


* 1024×1024	- $0.020

**Test image generation**
"""

import cv2
import json
import os
from base64 import b64decode
from google.colab.patches import cv2_imshow
import openai
from base64 import b64decode
from pathlib import Path
from torch import autocast
from PIL import Image
import spacy
import pandas as pd
import torch
from torchtext import data
#We use the utils from COSMOS to load the test.json file. CD to COSMOS folder or add COSMOS.utils.config and COSMOS.utils.common_utils to get right path
from COSMOS.utils.config import DATA_DIR
from COSMOS.utils.common_utils import read_json_data
from COSMOS.utils.dataset_utils import modify_caption_replace_entities
from nltk.stem.snowball import SnowballStemmer
# python3 -m spacy download en
spacy_en = spacy.load('en_core_web_sm')
stemmer = SnowballStemmer(language="english")
from torchtext.vocab import GloVe, FastText
from config import *

#Test function for a single prompt to check if your API connection works
"""
PROMPT = "Student drinking coffee and writing a master thesis"
DATA_DIR = Path.cwd() / "responses"

DATA_DIR.mkdir(exist_ok=True)

openai.api_key = OPENAI_API_KEY

#request image from Dall-E
response = openai.Image.create(
    prompt=PROMPT,
    n=1,
    size="512x512",
    response_format="b64_json",
)

#save JSON response file
file_name = DATA_DIR / f"{PROMPT[:5]}-{response['created']}.json"

with open(file_name, mode="w", encoding="utf-8") as file:
    json.dump(response, file)

#The script then fetches the Base64-encoded string from the JSON data,
#decodes it, and saves the resulting image data as a PNG file in a directory. 


DATA_DIR = Path.cwd() / "responses"
JSON_FILE =file_name
IMAGE_DIR = Path.cwd() / "images" / JSON_FILE.stem

IMAGE_DIR.mkdir(parents=True, exist_ok=True)

with open(JSON_FILE, mode="r", encoding="utf-8") as file:
    response = json.load(file)

for index, image_dict in enumerate(response["data"]):
    image_data = b64decode(image_dict["b64_json"])
    image_file = IMAGE_DIR / f"{JSON_FILE.stem}-{index}.png"
    print(image_file)
    with open(image_file, mode="wb") as png:
        png.write(image_data)

    #show the saved image in Colab 
    im = cv2.imread(str(image_file))
    cv2_imshow(im)

"""

"""
===============
If you run into an error, reset position of test_data.json caption to read and count to the same position
==============
"""


def DALLE2_gen():
    #Add desired data_dir path for JSON file response dump
    DATA_DIR = Path.cwd() / "NewDatastes/DALL-E2/responses"
    DATA_DIR.mkdir(exist_ok=True)
    #Add desired image dir path where images should be saved
    IMAGE_DIR = Path.cwd() / "NewDatastes/DALL-E2/images"
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)

    openai.api_key = OPENAI_API_KEY

    count = 0
    # Generate specific images
    with open(os.path.join(DATA_DIR, 'test_data.json')) as f:
        test_data = [json.loads(line) for line in f][0:5]

        for i in test_data:
            description_1 = i['caption1']
            description_2 = i['caption2']

            #NER tagging. We use the same code and model from COSMOS to ensure comparability.
            # You can also just load the caption1_modified and caption2_modified
            description_1 = modify_caption_replace_entities(description_1)
            description_2 = modify_caption_replace_entities(description_2)

            #Censoring
            #We split words on comma delimiter. This way we can add conconated words to the list
            with open('badwords.txt','r') as f:
                for line in f:
                    for word in line.split(","):
                        description_1 = description_1.replace(word, ''*len(word))
                        description_2 = description_2.replace(word, ''*len(word))

            #Print captions to see which image is being generated 
            print(description_1)
            print(description_2)

            #request image1 from Dall-E
            response1 = openai.Image.create(
                prompt=description_1,
                n=1,
                size="512x512",
                response_format="b64_json",
            )

            #request image2 from Dall-E
            response2 = openai.Image.create(
                prompt=description_2,
                n=1,
                size="512x512",
                response_format="b64_json",
            )

            #save JSON response files
            file_name1 = DATA_DIR / f"{count}_gen1.json"
            file_name2 = DATA_DIR / f"{count}_gen2.json"


            with open(file_name1, mode="w", encoding="utf-8") as file:
                json.dump(response1, file)
            with open(file_name2, mode="w", encoding="utf-8") as file:
                json.dump(response2, file)



            #The script then fetches the Base64-encoded string from the JSON data,
            #decodes it, and saves the resulting image data as a PNG file in a directory. 

            with open(file_name1, mode="r", encoding="utf-8") as file1:
                response1 = json.load(file1)

            for index1, image_dict1 in enumerate(response1["data"]):
                image_data1 = b64decode(image_dict1["b64_json"])
                image_file1 = IMAGE_DIR / f"{count}_gen1.png"
                print(image_file1)
                with open(image_file1, mode="wb") as png1:
                    png1.write(image_data1)

            with open(file_name2, mode="r", encoding="utf-8") as file2:
                response2 = json.load(file2)

            for index2, image_dict2 in enumerate(response2["data"]):
                image_data2 = b64decode(image_dict2["b64_json"])
                image_file2 = IMAGE_DIR / f"{count}_gen2.png"
                print(image_file2)
                with open(image_file2, mode="wb") as png2:
                    png2.write(image_data2)
            
                # +1 count so filename corresponds with original image position in test_data.json
            count = count + 1

