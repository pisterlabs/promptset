import requests
import json
import os
import openai

news_key = os.environ['newsapi_key']
country = "us"
news_url = f"https://newsapi.org/v2/top-headlines?country={country}&apiKey={news_key}"
news_result = requests.get(news_url)
news_data = news_result.json()

headlines = [article['title'] for article in news_data['articles']]
urls = [article['url'] for article in news_data['articles']]

openai.api_key = os.environ['openai_key']

prompt = "\n".join(headlines)

response = openai.Completion.create(
  model="text-davinci-002",
  prompt=prompt,
  temperature=0.5,
  max_tokens=150
)

for i in range(5):
  print(f"\n{headlines[i]}")
  print(urls[i])

  current_summary = response["choices"][i]["text"].strip()
  print(f"Summary: {current_summary}\n")