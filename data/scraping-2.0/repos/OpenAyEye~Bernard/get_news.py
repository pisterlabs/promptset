import os
import json
import requests
import boto3
from pydub import AudioSegment
from pydub.playback import play
from dotenv import load_dotenv
from eventregistry import EventRegistry, QueryArticlesIter, QueryItems
from datetime import datetime
import openai
import textwrap
import random
from openai import OpenAI

# Load environment variables from the .env file
load_dotenv("config.env")
# Access the OpenAI key from the environment variable
openai.api_key = os.environ.get("OpenAiKey")
# Load environment variables from the .env file
client = OpenAI()

# Access the OpenAI key from the environment variable
api_key = os.environ.get("news_api_key")
# Set up AWS Polly client
polly_client = boto3.client("polly", region_name="us-west-2")

# Function to convert text to speech using AWS Polly

def convert_text_to_speech(text):
    response = polly_client.synthesize_speech(
        Text=text,
        OutputFormat="mp3",
        VoiceId="Matthew"
    )

    audio_stream = response["AudioStream"].read()

    # Save the synthesized speech as an MP3 file
    with open("news_update.mp3", "wb") as file:
        file.write(audio_stream)

    # Load the audio file using PyDub
    audio = AudioSegment.from_mp3("news_update.mp3")

    # Play the audio
    play(audio)

    # Delete the MP3 file
    os.remove("news_update.mp3")

# Function to fetch news updates from the News API





def fetch_articles(api_key):
    er = EventRegistry(apiKey=api_key)

    # URIs for major categories
    business_uri = er.getCategoryUri("business")
    tech_uri = er.getCategoryUri("tech")
    entertainment_uri = er.getCategoryUri("entertainment")
    health_uri = er.getCategoryUri("health")
    science_uri = er.getCategoryUri("science")
    politics_uri = er.getCategoryUri("politics")

    # Query for articles in English from the USA, excluding sports-related articles
    q = QueryArticlesIter(
        locationUri=er.getLocationUri("United States"),
        lang="eng",
        categoryUri=QueryItems.OR([business_uri, tech_uri, entertainment_uri, health_uri, science_uri, politics_uri]),
        ignoreKeywords=QueryItems.OR(['sports', 'sport'])
    )

    articles = []
    for art in q.execQuery(er, sortBy="date", maxItems=25):  # Fetch top 25 articles
        # Check if article with the same title is already in the list
        if art['title'] not in [a['title'] for a in articles]:
            articles.append(art)

    # Select 5 random articles from the top 25 articles
    random_articles = random.sample(articles, 5)

    return random_articles




def chat_completion_request(user_input):

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {
                "role": "system",
                "content": "You are a useful assistant."
            },
            {
                "role": "user",
                "content": f"{user_input}"
            }
        ],
        temperature=0.9,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=1,
        stop=["\nUser:"],
    )

    # Get the response content
    response_content = response.choices[0].message.content#"choices"][0]["message"]["content"]

    # Add line breaks every 120 characters or at the nearest space to 120 characters
    wrapped_response_content = textwrap.fill(response_content, width=120)

    return response_content # wrapped_response_content

def chat_completion_request_bak(user_input):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613",
        messages=[
            {
                "role": "system",
                "content": "You are a useful assistant."
            },
            {
                "role": "user",
                "content": f"{user_input}"
            }
        ],
        temperature=0.9,
        max_tokens=1000,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        n=1,
        stop=["\nUser:"],
    )
    return response["choices"][0]["message"]["content"]
def main():
    # Fetch the latest news updates

    news_updates = fetch_articles(api_key)
    # Read each news update out loud

    for news_update in news_updates:
        date = datetime.strptime(news_update["date"], '%Y-%m-%d')  # Convert date string to datetime object
        url = news_update["url"]
        date_str = date.strftime("%Y-%m-%d")  # Format date as string
        description = chat_completion_request(f"please summarize the following news article in a few sentences: '{news_update['body']}'")
        from time import sleep
        sleep(1)
        print("*****************************************************************************************************")
        print(f"Date: {date_str} -'{news_update['title']}'")
        print("-----------------------------------------------------------------------------------------------------")
        wrapped_description = textwrap.fill(description, width=120)
        print(wrapped_description)
        print("_____________________________________________________________________________________________________")
        print("")
        print("")

        headlines = f"'{news_update['title']}' {description}"
        #
        #convert_text_to_speech(headlines)
    return headlines
if __name__ == "__main__":

    main()