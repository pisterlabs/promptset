from openai import OpenAI
from instagrapi import Client
from dotenv import load_dotenv

import base64
import os
import tempfile
import logging
import uuid

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s')

#OpenAI API Key and Insta creds need to be stored in environment variables or .env file
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
INSTA_USER = os.environ.get("KIDS_USER")
INSTA_PASSWORD = os.environ.get("KIDS_PASSWORD")

OpenAI_Client = OpenAI(api_key=OPENAI_API_KEY)

def createPostImagePrompt(testPrompt=False):
  if (not testPrompt):
    #Use standard prompt
    logging.info("Preparing Insta image prompt...")
    promptMessages=[
        {"role": "system", "content": """You are a four year old girl who comes up with silly and creative Dall-e prompts to make fun pictures."""},
        {"role": "user", "content": """Give me a prompt without quotation marks."""}
      ]
  else:
    #Use test prompt
    logging.info("Preparing Insta image prompt (test prompt)...")
    promptMessages=[
        {"role": "system", "content": """You are a four year old girl who comes up with silly and creative Dall-e prompts to make fun pictures."""},
        {"role": "user", "content": """Give me a prompt without quotation marks."""}
      ]

  text_prompt = OpenAI_Client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=promptMessages
  )

  return text_prompt.choices[0].message.content

def createPostImage(picture_prompt, testPrompt=False):
  logging.info("Preparing Insta image...")
  picture_response = OpenAI_Client.images.generate(
    model="dall-e-3",
    prompt=picture_prompt,
    size="1024x1024",
    quality="hd",
    response_format="b64_json",
    n=1,
  )

  return picture_response.data[0]

def createPostCaption(picture_prompt, testPrompt=False):
  post_caption = picture_prompt + " #ai #aiart #kidsart"

  return post_caption

def saveImage(b64_json):
  image_filename = os.path.join(tempfile.gettempdir(),"kidsart-" + str(uuid.uuid4().hex) + ".jpg")

  with open(image_filename, "wb") as fh:
    fh.write(base64.b64decode(b64_json))
    logging.info("Image saved: " + image_filename)

  return image_filename


def postToInsta(image_filename, post_caption):
  logging.info("Posting to Instagram...")
  cl = Client()
  cl.login(INSTA_USER, INSTA_PASSWORD)
  cl.photo_upload(path=image_filename, caption=post_caption)
  logging.info("Instagram post completed!")

def makeKidsArt():
  logging.info("Making Kids Art")
  #Get A 4 Year Old Girl Prompt
  picture_prompt = createPostImagePrompt()
  #picture_prompt_test = createPostImagePrompt(True)

  #Get Image Based on Summary
  picture_response = createPostImage(picture_prompt)
  #picture_response_test = createPostImage(picture_prompt, True)

  #Save Image to TMP
  image_filename = saveImage(picture_response.b64_json)
  #image_filename_test = saveImage(picture_response_test.b64_json)

  #Get Caption for Post
  post_caption = createPostCaption(picture_prompt)
  #post_caption_test = createPostCaption(picture_prompt, True)

  #Post to Instagram
  postToInsta(image_filename, post_caption)

if __name__ == '__main__': 
  makeKidsArt() 