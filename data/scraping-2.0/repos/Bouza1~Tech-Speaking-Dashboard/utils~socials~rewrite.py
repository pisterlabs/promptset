import os
import openai
from dotenv import load_dotenv
import json

load_dotenv('deets.env')

def rewrite_4_socials(article):
  openai.api_key = os.getenv('AI_API_KEY')
  prompt = "Summarise the text below, like a post you would find on any reputable technology-based social media account. Try and limit the summary to just one paragraph. " + article
  completion = openai.ChatCompletion.create(model="gpt-3.5-turbo",
                                            messages=[{
                                              "role": "user",
                                              "content": prompt
                                            }],
                                            max_tokens=2000,
                                            temperature=0.8)
  return completion['choices'][0]['message']['content']

def append_rewite_2_live_articles(json_url, id, post):
  with open(json_url) as openfile:
    full_json_object = json.load(openfile)
    news_articles = full_json_object['news']
    for article in news_articles:
      if article['id'] == id:
        article.update({'social_post_text':post})
  with open(json_url, 'w') as json_file:
    json.dump(full_json_object, json_file, indent=4,  separators=(',',': '))
