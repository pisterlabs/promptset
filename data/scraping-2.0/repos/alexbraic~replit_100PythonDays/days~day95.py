# get the latest 5 news from newsAPI
# summarize each article content in 2-3 words using openAi
# in Spotify API, search for tracks that have similar names
#+to the article summary
# show the title and preview_url for 1 track for each summary

import requests, json, os
import openai
from requests.auth import HTTPBasicAuth

# secrets kept in Repl
API_KEY = os.environ['API_KEY']

# openai auth (expired credits so will not generate summary)
openai.apikey = os.environ['openAI']
openai.organization = os.environ['organizationID']
openai.Model.list()

# newsAPI ===========================================

# send the requests and get the latest nes
country = "us"
url = f"https://newsapi.org/v2/top-headlines?country={country}&apiKey={API_KEY}"

response = requests.get(url)
articles = response.json()
#print(json.dumps(articles, indent=2))

i = 0
for article in articles["articles"]:
  if i == 5:
    break
  #print(article["title"])
  #print(article["content"])
  #print()
  i += 1

  # open ai ===========================================

  # create openAi task and url
  prompt = f'Summarize the following article in 2 or 3 words.\n {article["content"]}'
  response = openai.Completion.create(model="text-davinci-002",
                                      prompt=prompt,
                                      temperature=0,
                                      max_tokens=6)
  # print the summary
  #print(response["choices"][0]["text"].strip())
  # use the summary to search for a track on Spotify
  summary = response["choices"][0]["text"].strip()

  # spotify ===========================================

  # instanciate the client auth variables
  client_id = os.environ['CLIENT_ID']
  client_secret = os.environ['CLIENT_SECRET']
  # build the post request that gets the access token
  get_token_url = "https://accounts.spotify.com/api/token"
  data = {'grant_type': 'client_credentials'}
  auth = HTTPBasicAuth(client_id, client_secret)

  # post request to get access token
  resp = requests.post(get_token_url, data=data, auth=auth)
  #print(resp.json())

  # access token and bearer
  access_token = resp.json()['access_token']
  headers = {'Authorization': f'Bearer {access_token}'}

  # build the url to search for tracks once auth is complete
  s_url = "https://api.spotify.com/v1/search"
  search = f"?q={summary}&type=track&limit=1"
  full_URL = f'{s_url}{search}'

  # save the response in a variable
  songs = requests.get(full_URL, headers=headers)
  songs_result = songs.json()

  # output the result main points: name and preview_url
  for track in songs_result["tracks"]["items"]:
    print(track["name"])
    print(track["preview_url"])
    print()
