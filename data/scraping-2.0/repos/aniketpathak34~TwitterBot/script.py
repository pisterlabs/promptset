import tweepy
import os
import openai
import random
import time
from datetime import datetime




def create_tweet():
    openai.organization = "enter account organization"
    openai.api_key = "enter your openai key"






# List of phrases and keywords

    trending_phrases_prompt=[
        "Describe the future of {choose any  topic by your self} in 50 words or less. Use hashtags to highlight key aspects, add one hashtag at the end, make all this under 50 words including hashtag, give me unique content dont copy it",
        "Imagine {choose any  topic by your self} as a character in a story. Provide a short backstory and the role it plays aspects,add one hashtag at the end, make all this under 50 words including hashtag, give me unique content dont copy it",
        "Share a surprising fact or statistic about {choose any  topic by your self}. Let's make it eye-opening aspects,add one hashtag at the end, make all this under 50 words including hashtag, give me unique content dont copy it",
        "Create a haiku poem inspired by {choose any  topic by your self}. Three lines, 5-7-5 syllable pattern aspects,add one hashtag at the end, make all this under 50 words including hashtag, give me unique content dont copy it",
        "Invent a new product or service related to {choose any  topic by your self}. Give it a catchy name and describe its benefits aspects,add one hashtag at the end, make all this under 50 words including hashtag, give me unique content dont copy it",
        "Explain the impact of {your_topic} on society and culture. Consider both positive and negative effects aspects,add one hashtag at the end, make all this under 50 words including hashtag, give me unique content dont copy it",
        "Compose a short advice or motivational message inspired by {choose any  topic by your self}. Share wisdom in just a few words aspects, add one hashtag at the end, make all this under 50 words including hashtag, give me unique content dont copy it",
        "If {choose any  topic by your self} could talk, what would it say? Write a fun and imaginative dialogue aspects,add one hashtag at the end, make all this under 50 words including hashtag, give me unique content dont copy it",
        "Craft a micro-story (40 characters or less) that involves {choose any  topic by your self}. Create intrigue in a few words aspects,add one hashtag at the end, make all this under 50 words including hashtag, give me unique content dont copy it",
        "Visualize {choose any  topic by your self} as a work of art. Describe the colors, shapes, and emotions it evokes aspects, add one hashtag at the end, make all this under 50 words including hashtag, give me unique content dont copy it"

    ]
   
    prompts = random.choice(trending_phrases_prompt)




    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # joke_with_timestamp = f"{joke} [{timestamp}]"



   

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt = prompts,
        max_tokens = 100,
        temperature = 0.5
    )
    joke = response['choices'][0]['text']
    tweet_text = joke[:280]
    print(joke)

#you will get this from twitter developer portal
    api_key = "enter key"
    api_secret_key = "enter secret key"
    bearer_token = r"enter bearer_token"
    access_token = "enter access_token"
    access_token_secret_key = "enter access_token_secret_key"


    client =  tweepy.Client(bearer_token, api_key, api_secret_key, access_token, access_token_secret_key)
    auth = tweepy.OAuth1UserHandler(api_key, api_secret_key, access_token, access_token_secret_key)
    api = tweepy.API(auth)


    client.create_tweet(text=tweet_text)

    print("Tweet created at:", timestamp)


# Run the program every 30 minutes
while True:
    create_tweet()
    time.sleep(1800)  # Sleep for 1800 seconds (30 minutes)
