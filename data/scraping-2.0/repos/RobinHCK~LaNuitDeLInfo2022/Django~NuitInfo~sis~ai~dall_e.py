# Imports
import os
import uuid
import requests

import openai
from PIL import Image, ImageDraw, ImageFont

OPENAI_KEY = "PUT-YOUR-OWN-KEY"
PATH_STATIC         = "./sis/static/"
PATH_LOGO_GENERATED = "logos_generated/"
PATH_LOGO_SIS		= "logo_sis/logo_sida_{}.png"

def create_DALL_E_image(sentence, text_overlay, size_image):
	""" Create a DALL-E image and return the URL

	Send a request to the openai API using the personal key.
	An image at a given size which match the sentence is generated.
	A URL link lead to the image.

	(str) sentence : The sentence used to generate the DALL-E image.
	(int) size_image : The image size (size x size). 
					   Must be 256x256 or 512x512 or 1024x1024
	(str) key : The openai API key (see https://openai.com/). 
		    Robin's KEY 
	"""
	openai.api_key = OPENAI_KEY
	
	if size_image != 256 and size_image != 512 and size_image != 1024:
		raise ValueError("The image size must be 256x256 or 512x512 or 1024x1024")

	response = openai.Image.create(
		prompt = sentence,
		n = 1,
		size = str(size_image) + "x" + str(size_image)
	)

	image_url = response['data'][0]['url']

	filename = save_image_from_url(image_url)
	add_description_on_image(text_overlay, filename)
	add_logo_on_image(size_image, filename)
	return filename

def save_image_from_url(image_url):
	""" Save the image of an URL to a local file 
	
	(str) image_url : The url of the image
	"""

	img_data = requests.get(image_url).content
	filename = os.path.join(PATH_LOGO_GENERATED, str(uuid.uuid4()) + ".png")

	with open(os.path.join(PATH_STATIC, filename), 'wb') as handler:
		handler.write(img_data)
	return filename


def add_logo_on_image(size_image, path_image):
	""" Add a logo on an image

	The image and the logo must be at the same size.
	
	(str) path_logo : The path of the logo.
	(str) path_image : The path of the image.
	(str) path_out : The path where the image with the logo will be saved.
	"""

	image = Image.open(os.path.join(PATH_STATIC, path_image)).convert('RGB')
	logo  = Image.open(os.path.join(PATH_STATIC, PATH_LOGO_SIS.format(size_image))).convert('RGBA')

	image.paste(logo,box=(0,0),mask=logo)
	image.save(os.path.join(PATH_STATIC,path_image))

def add_description_on_image(description, path_image):
    """ Add a funny description on an image

    (str) description : The description message to put on the image.
    (str) path_image : The path of the image.
    (str) path_out : The path where the image with the description will be saved.
    """
    path_image = os.path.join(PATH_STATIC, path_image)
    image = Image.open(path_image).convert('RGB')

    draw = ImageDraw.Draw(image)
    # Arial path needs to be changed, but fonts works differently on windows and Ubuntu
    draw.text((10, image.size[0]-50), description, font=ImageFont.truetype("/home/gautier/.local/share/fonts/arial.ttf", size=25), fill="black")

    image.save(path_image)
