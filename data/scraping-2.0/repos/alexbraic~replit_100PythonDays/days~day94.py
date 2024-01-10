# get the latest 5 news of the day and create a summary using openAi
# create news API/openAi account and get api key
# send request to newsAPI and output the first 5 articles
# create a summarise url for openAi and output the summary

import requests, json, os
import openai

# auth
# in Replit, the keys are saved as secrets
# in other projects, the keys can be held and referenced in
#+a different file
API_KEY = os.environ['API_KEY']  # newsAPI
openai.apikey = os.environ['openAI']  # openAi
openai.organization = os.environ['organizationID']  # openAi
openai.Model.list()

# change country to get news articles from different ones
country = "ie"
# create request url
url = f"https://newsapi.org/v2/top-headlines?country={country}&apiKey={API_KEY}"

# send request to newsAPI
result = requests.get(url)
data = result.json()
print(json.dumps(data, indent=2))  # print this to check result

# loop through articles and get main info for first 5 articles
i = 1
for article in data["articles"]:
  if i == 6:
    break
  print(f'\t{i}: {article["title"]}')
  print(article["url"])
  print(article["content"])

  # create openAi task and url
  prompt = f'Summarize the following article {article["content"]}'
  response = openai.Completion.create(model="text-davinci-002",
                                      prompt=prompt,
                                      temperature=0,
                                      max_tokens=6)
  # print the summary
  print(response["choices"][0]["text"].strip())
  i += 1
