from openai import OpenAI
from instagrapi import Client
from dotenv import load_dotenv

import base64
import os
import tempfile
import logging
import uuid
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s')

#OpenAI API Key and Insta creds need to be stored in environment variables or .env file
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
INSTA_USER = os.environ.get("FOOD_USER")
INSTA_PASSWORD = os.environ.get("FOOD_PASSWORD")

OpenAI_Client = OpenAI(api_key=OPENAI_API_KEY)

def chooseMeal():
  logging.info("Choosing meal...")
  current_hour = datetime.now(timezone.utc).hour
  
  if (current_hour > 10 and current_hour < 17):
    return "breakfast"
  elif (current_hour > 16 and current_hour < 23):
    return "lunch"
  elif (current_hour == 23 or current_hour < 5):
    return "dinner"
  else:
    return "snack"

def createPostImage(meal, testPrompt=False):
  logging.info("Preparing Insta image...")
  picture_response = OpenAI_Client.images.generate(
    model="dall-e-3",
    prompt="Show any delicious " + meal + " meal from a random nationality as a high resolution, highly realistic, highly detailed photograph with a Sony a7R III.",
    size="1024x1024",
    quality="hd",
    response_format="b64_json",
    n=1,
  )

  return picture_response.data[0]

def createPostCaption(revised_picture_prompt, meal, testPrompt=False):
  if (not testPrompt):
    #Use standard prompt
    logging.info("Preparing Insta caption...")
    postMessages = [
      #{"role": "system", "content": """You run an Instagram account for food photography."""},
      {"role": "user", "content": """Describe the food in image in less than 300 characters. Do not wrap the caption in quotation marks.\n"""+revised_picture_prompt}
    ]
  else:
    #Use test prompt
    logging.info("Preparing Insta caption (test prompt)...")
    postMessages = [
      #{"role": "system", "content": """You run an Instagram account for food photography."""},
      {"role": "user", "content": """Describe the food in image in less than 300 characters. Do not wrap the caption in quotation marks.\n"""+revised_picture_prompt}
    ]

  text_prompt = OpenAI_Client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=postMessages
  )

  #post_caption = "Today's " + meal + " ... "
  post_caption = text_prompt.choices[0].message.content + " #food #foodie #foodphotography #foodporn #foodstagram #foodies #foodlover #foodpics"

  return post_caption

def saveImage(b64_json):
  image_filename = os.path.join(tempfile.gettempdir(),"foodart-" + str(uuid.uuid4().hex) + ".jpg")

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

def makeFoodArt():
  logging.info("Making Food Art")

  #Get Current Meal
  meal = chooseMeal()

  #Get Image Based on Prompt
  picture_response = createPostImage(meal)
  #picture_response_test = createPostImage(meal, True)

  #Save Image to TMP
  image_filename = saveImage(picture_response.b64_json)
  #image_filename_test = saveImage(picture_response_test.b64_json)

  #Get Caption for Post
  post_caption = createPostCaption(picture_response.revised_prompt, meal)
  #post_caption_test = createPostCaption(picture_response.revised_prompt, meal, True)

  #Post to Instagram
  postToInsta(image_filename, post_caption)

if __name__ == '__main__': 
  makeFoodArt() 