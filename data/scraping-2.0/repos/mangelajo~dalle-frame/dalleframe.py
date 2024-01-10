#!/usr/bin/python
# -*- coding:utf-8 -*-
import logging
import os
import requests
import sys
import traceback
import time

from waveshare_epd import epd5in65f
from PIL import Image,ImageDraw,ImageFont

from openai import OpenAI

client = OpenAI(api_key=os.getenv('OPENAI_KEY'))

IMAGE_FOLDER="static/images/"


def request_image(prompt):
    # Set your API key

    print("Requesting image from DALL-E with prompt: ", prompt)
    # Generate the image with DALL-E
    response = client.images.generate(prompt=prompt, model="dall-e-3", n=1, size="1024x1024")

    # Get the URL of the generated image
    image_url = response.data[0].model_dump()['url']

    filename = "dall-e-image-" + time.strftime("%Y%m%d-%H%M%S") + "_" + prompt.replace(' ', '-') + ".png"
    image_path = os.path.join(IMAGE_FOLDER, filename)

    # Download the generated image
    response = requests.get(image_url)
    with open(image_path, "wb") as f:
        f.write(response.content)

    return image_path

def display_eink_image(image_path):

    EPD_WIDTH = 600
    EPD_HEIGHT = 448

    #define palette array
    palettedata = [
            0, 0, 0,
            255, 255, 255,
            67, 138, 28,
            100, 64, 255,
            191, 0, 0,
            255, 243, 56,
            232, 126, 0,
            194 ,164 , 244
        ]
    p_img = Image.new('P', (16, 16))
    p_img.putpalette(palettedata * 32)

    print("Converting image to e-ink format")
    # Open the image with PIL
    image = Image.open(image_path)
    resized_img = image.resize((EPD_WIDTH, EPD_WIDTH))
    # we need to crop a bit on the height to fit the screen
    cropped_img = resized_img.crop((0, 32, EPD_WIDTH, 32+EPD_HEIGHT))
    colored_img = cropped_img.quantize(palette=p_img)

    epd = epd5in65f.EPD()
    epd.init()

    print("Drawing image")
    epd.display(epd.getbuffer(colored_img))

    epd.sleep()




