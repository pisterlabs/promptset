import random
from interface import Message
from dotenv import load_dotenv
import csv 
import os
import tweepy
import collections
import json
import openai

class Scraper:
    def __init__(self, interfaceObject, listOfAccounts):
        self.program = interfaceObject
        self.listOfAccounts = listOfAccounts
        load_dotenv()
        # Set your API keys and tokens
        consumer_key = os.environ.get("CONSUMER_KEY")
        consumer_secret = os.environ.get("CONSUMER_SECRET")
        access_token = os.environ.get("ACCESS_TOKEN")
        access_token_secret = os.environ.get("ACCESS_TOKEN_SECRET")
        print(consumer_key, consumer_secret, access_token, access_token_secret)
        # Authenticate with the Twitter API
        auth = tweepy.OAuth1UserHandler(consumer_key, consumer_secret, access_token, access_token_secret)
        # Create a Tweepy API object
        self.api = tweepy.API(auth, timeout=120)
        # Store the tweets in a dictionary - user : [tweets]
        self.user_tweets = collections.defaultdict(list)

    def run(self):
        print("Scraper running")
        for account in self.listOfAccounts:
            user_timeline = self.api.user_timeline(screen_name=account, count=10, tweet_mode='extended')
            for tweet in user_timeline:
                if hasattr(tweet, 'retweeted_status'):
                    # If it's a retweet, store the retweeted_status object instead
                    self.user_tweets[account].append(tweet.retweeted_status)
                else:
                    self.user_tweets[account].append(tweet)

    def getSinglePrediction(self, content) -> str:
        openai.api_key = os.environ.get("OPENAI_API_KEY")
        lstOfCategories = ['web3 activities and events', 'web3 announcements', 'web3 research output', 'web3 meme', 'crypto and markets', 'web3 phishing or irrelevant', 'unknown']

        response = openai.Completion.create(
            model="text-davinci-003",
            prompt=f"Decide which category the Tweet best classify as {lstOfCategories}.\n\nTweet: \"{content}\"\nCategory: ",
            temperature=0,
            max_tokens=60,
            top_p=1,
            frequency_penalty=0.5,
            presence_penalty=0
            )

        try:
            result = response['choices'][0]['text']
            print('Prediction: ', result[-len(result)+1:])
            return result[-len(result)+1:]
        except:
            return f"Error: {response}"

    def save_tweets_to_csv(self):
        # Save the tweet's text to a csv file for chatGPT's usage 
        print("Saving tweets to csv")
        with open(f'tweets/tweets_text_only.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['user','tweet_content'])
            for account in self.user_tweets:
                for tweet in self.user_tweets[account]:
                    writer.writerow([account, tweet.full_text])
        
        posts = []
        for account in self.user_tweets:
            for tweet in self.user_tweets[account]:
                post = {
                    "displayName": tweet.user.name,
                    "username": tweet.user.screen_name,
                    "avatar": tweet.user.profile_image_url_https,
                    "verified": tweet.user.verified,
                    "image": None,
                    "text": tweet.full_text,
                    "label": self.getSinglePrediction(tweet.full_text),
                }

                if 'media' in tweet.entities:
                    for media in tweet.entities['media']:
                        if media['type'] == 'photo':
                            post['image'] = media['media_url_https']
                            break

                posts.append(post)
        # shuffle the list posts
        random.shuffle(posts)
        # save the tweet's text and media to a csv file for frontend usage
        with open('frontend/src/data/posts_with_labels.json', 'w') as file:
            json.dump(posts, file, indent=4)
