from openai import OpenAI
from gnews import GNews
from instagrapi import Client
from dotenv import load_dotenv

import random
import base64
import os
import tempfile
import logging
import uuid

#OpenAI API Key and Insta creds need to be stored in environment variables or .env file
load_dotenv()
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
INSTA_USER = os.environ.get("INSTA_USER")
INSTA_PASSWORD = os.environ.get("INSTA_PASSWORD")

OpenAI_Client = OpenAI(api_key=OPENAI_API_KEY)

def getNewsArticle():
  logging.info("Fetching top news article...")
  google_news = GNews(language='en', period='1d', max_results=10)
  google_news.exclude_websites = ['reuters.com','thehill.com','ft.com']
  top_news = google_news.get_top_news()

  chosen_news = random.choice(top_news)

  article = google_news.get_full_article(chosen_news['url'])

  logging.info("Article chosen: " + article.title + "\n" + article.url)

  return article

def createPostImagePrompt(article, testPrompt=False):
  if (not testPrompt):
    #Use standard prompt
    logging.info("Preparing Insta image prompt...")
    promptMessages=[
        {"role": "system", "content": """You are an assistant whose role is to read news articles and 
        provide rich details about that article so that a painter can create an artistic representation 
        of it."""},
        {"role": "user", "content": """Summarize the following news article with as much vivid detail as possible.
        \n\n"""+article.title + "\n" + article.text}
      ]
  else:
    #Use test prompt
    logging.info("Preparing Insta image prompt (test prompt)...")
    promptMessages=[
        {"role": "system", "content": """You are an assistant whose role is to read news articles and 
        provide rich details about that article so that a painter can create an artistic representation 
        of it."""},
        {"role": "user", "content": """Summarize the following news article with as much vivid detail as possible.
        \n\n"""+article.title + "\n" + article.text}
      ]

  text_prompt = OpenAI_Client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=promptMessages
  )

  return text_prompt.choices[0].message.content

def createPostImage(article_summary, testPrompt=False):
  if (not testPrompt):
    #Use standard prompt
    logging.info("Preparing Insta image...")
    art_prompt = """Create a painting in any style based on the following news article summary. If the article is inappropriate, 
    make it appropriate enough for Dall-E. Highly detailed. 8K. No text.\n"""
  else:
    #Use test prompt
    logging.info("Preparing Insta image (test prompt)...")
    art_prompt = """Create a painting in any style based on the following news article summary. If the article is inappropriate, 
    make it appropriate enough for Dall-E. Highly detailed. 8K. No text.\n"""

  art_prompt += article_summary

  picture_response = OpenAI_Client.images.generate(
    model="dall-e-3",
    prompt=art_prompt,
    size="1024x1024",
    quality="hd",
    response_format="b64_json",
    n=1,
  )

  return picture_response.data[0]

def createPostCaption(revised_prompt, article, testPrompt=False):
  if (not testPrompt):
    #Use standard prompt
    logging.info("Preparing Insta caption...")
    postMessages = [
      {"role": "system", "content": """You run an Instagram account for news inspired artwork, 
     are an expert on social media marketing, and in your late twenties with a college degree."""},
      {"role": "user", "content": """Provide a short, but engaging Instagram post caption for 
     artwork created by the following prompt about a current event. Be sure to include hashtags, 
     include the art style used, and include emojis. This caption should be ready to post as is - 
     do not try to give credit to the artist or add a link to another website. Do not mention any 
     other Instagram account. Do not mention that the picture is 8K. Do not wrap the caption 
       in quotation marks or proceed it with 'Caption:'. Keep it under 500 characters.\n"""+revised_prompt}
    ]
  else:
    #Use test prompt
    logging.info("Preparing Insta caption (test prompt)...")
    postMessages = [
      {"role": "system", "content": """You run an Instagram account for news inspired artwork, 
     are an expert on social media marketing, and in your late twenties with a college degree."""},
      {"role": "user", "content": """Provide a short, but engaging Instagram post caption for 
     artwork created by the following prompt about a current event. Be sure to include hashtags, 
     include the art style used, and include emojis. This caption should be ready to post as is - 
     do not try to give credit to the artist or add a link to another website. Do not mention any 
     other Instagram account. Do not mention that the picture is 8K. Do not wrap the caption 
       in quotation marks or proceed it with 'Caption:'. Keep it under 500 characters.\n"""+revised_prompt}
    ]

  text_prompt = OpenAI_Client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=postMessages
  )

  post_caption = "\U0001f5BC  Inspired by latest top news: " + article.title + "\n\n"
  post_caption += text_prompt.choices[0].message.content

  return post_caption

def saveImage(b64_json):
  image_filename = os.path.join(tempfile.gettempdir(),"newsart-" + str(uuid.uuid4().hex) + ".jpg")

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

def makeNewsArt():
  logging.info("Making News Art")
  #Get News Article
  article = getNewsArticle()

  #Get Summary of Article
  article_summary = createPostImagePrompt(article)
  #article_summary_test = createPostImagePrompt(article, True)

  #Get Image Based on Summary
  picture_response = createPostImage(article_summary)
  #picture_response_test = createPostImage(article_summary, True)

  #Save Image to TMP
  image_filename = saveImage(picture_response.b64_json)
  #image_filename_test = saveImage(picture_response_test.b64_json)

  #Get Caption for Post
  post_caption = createPostCaption(picture_response.revised_prompt, article)
  #post_caption_test = createPostCaption(picture_response.revised_prompt, article, True)

  #Post to Instagram
  postToInsta(image_filename, post_caption)

if __name__ == '__main__': 
  makeNewsArt() 