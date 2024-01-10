from django.http.response import JsonResponse
from rest_framework.response import Response
import logging
from .serializer import *
from logtail import LogtailHandler
from datetime import datetime
import openai
import weaviate
import os
from memeModel import idl
from PIL import Image, ImageDraw, ImageFont


import base64
import io

# handler = LogtailHandler(source_token="tvoi6AuG8ieLux2PbHqdJSVR")
# logger = logging.getLogger(__name__)
# logger.handlers = []
# logger.addHandler(handler)
# logger.setLevel(logging.INFO)

WEAVIATE_URL = os.getenv('WEAVIATE_URL')
CLIENT_CONFIG = None

if WEAVIATE_URL:
    # If a URL is provided, assume this is our remote instance
    # and check for an access token.
    access_token = os.getenv('ACCESS_TOKEN')
    if not access_token:
        raise ValueError("ACCESS_TOKEN environment variable not set. " \
            "This is needed to log into the Weaviate instance.")
    CLIENT_CONFIG = weaviate.AuthBearerToken(
        access_token=access_token,
        expires_in=300 # this is in seconds, by default 60s
        )
else:
    WEAVIATE_URL = 'http://localhost:8080'

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is required for uploading objects to Weaviate")


CLIENT = weaviate.Client(WEAVIATE_URL,
                         auth_client_secret=CLIENT_CONFIG,
                         additional_headers={"X-OpenAI-Api-Key": OPENAI_API_KEY})

def construt_pil_img(contents):
    """
    Helper function for handling image
    """

    image_data = io.BytesIO(base64.decodebytes(contents.encode('ascii')))
    image = Image.open(image_data)
    if image.mode != "RGB":
        image = image.convert(mode="RGB")

    return image


def generate_text_for_meme(generateTextForMemeRequest):
  """
    Generate text for meme image
  """
  # This should take the input text and the image that corresponds to it and generate text
  # for the meme
  # Use GPT to generate the emotional description of the text
  # Use GPT to generate the emotional description of the text
  try:
    emotionalDescription = "Generate text for a meme about " + str(generateTextForMemeRequest.InputText) + " using standard meme templates"

    memeText = openai.Completion.create(
      engine="text-davinci-003",
      prompt=emotionalDescription,
      temperature=0.7,
      max_tokens=300,
      n=1,
      stop=None,
      frequency_penalty=0,
      best_of=1,
      presence_penalty=0,
    )

    # logger.info("Meme input text: " + str(memeText))
    # logger.info("Meme text: " + str(memeText.choices[0].text))

    # Return the meme URL and description
    return GenerateTextForMemeResponse(
      OutputText=memeText.choices[0].text,
      Error=None,
    )

  except Exception as e:
    return GenerateTextForMemeResponse(
      OutputText="",
      Error=str(e),
    )



def generate_meme_image(generateMemeImageRequest):
  """
    Add the generated meme text to the image response
      MemeImage: str
      MemeText: str
  """

  try:
    # Load the image

    print(f'Generating meme image for {generateMemeImageRequest.MemeText}')
    curPath = os.getcwd()
    image = construt_pil_img(generateMemeImageRequest.MemeImage)

    # image = Image.open(decoded_img)
    # image = Image.open(curPath + '/memeGeneration/query-result-image-0.png')

    print("Read in image. Positioning image")

    # Create a drawing context and set the font
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("Andale Mono.ttf", size=32)

    # Define the text to draw and its position
    text = generateMemeImageRequest.MemeText
    print(f"Image (width,height): {image.width},{image.height}")

    position = (50, 50)

    # Draw the text on the image
    draw.text(position, text, fill=(255, 255, 255), font=font)

    # Save the modified image to disk
    path = curPath + '/memeGeneration/meme_processed.png'
    image.save(curPath + '/memeGeneration/meme_processed.png')

    return GenerateMemeImageResponse(
      ProcessedMeme=path,
      Error=None,
    )

  except Exception as e:
    return GenerateMemeImageResponse(
      ProcessedMeme="",
      Error=str(e),
    )

