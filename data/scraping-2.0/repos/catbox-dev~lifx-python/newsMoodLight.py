# Description: This script gets the top news from the US and decides the sentiment of one random article, using the
#              openAI API for the sentiment analysis, the lifx API to change the brightness of the lights and the
#              newsAPI to get the news.

# Import the libraries
import requests
import openai
import configparser
import random

# Read the config file
config = configparser.ConfigParser()
config.read('config.ini')

# Get the API key from the config file
openai.api_key = config['openAI']['api_key']
newsAPI = config['newsAPI']['api_key']
lifxAPI = config['lifx']['api_key']

# Authenticate with the lifx API
headers = {
    "Authorization": f"Bearer {lifxAPI}",
}

# Make the url for the newsAPI request
newsURL = ('https://newsapi.org/v2/top-headlines?'
           'country=us&'
           f'apiKey={newsAPI}')

# Get the newsAPI response
response = requests.get(newsURL)

# Get the index of the article to be analyzed
index = random.randint(0, 15)   # also you can use the article's index that you want to analyze

# Assign the title and description of the first article to variables
title = response.json()['articles'][index]['title']
description = response.json()['articles'][index]['description']

# Prompt that decide the sentiment of the following news article
promptChat = ("Decide the sentiment of the following news article, you can only answer with decimal numbers from "
              "0 to 1, remember that 0 is negative and 1 is positive and no words.\n\n"
              f"Article's title: \"{title}\"\n"
              f"Article's description: \"{description}\"\n"
              "Sentiment in decimal numbers from 0 to 1: ")

# Get the openAI response
chatResponse = openai.Completion.create(
    model="text-davinci-003",
    prompt=promptChat,
    temperature=0.9,
    max_tokens=60,
    top_p=1.0,
    frequency_penalty=0.5,
    presence_penalty=0.0
)

# Print the title, description and sentiment
print(f"Article's title: {title}\n"
      f"Article's description: {description}\n"
      f"Sentiment in decimal numbers from 0 to 1: {chatResponse['choices'][0]['text']}")

# Set the payload
payload = {
    "power": "on",
    "brightness": chatResponse['choices'][0]['text']
}

# Get the response
response = requests.put('https://api.lifx.com/v1/lights/all/state', data=payload, headers=headers)