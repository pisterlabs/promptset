import datetime
import requests
import pandas as pd
import json
import ast
import yaml
import openai

def process_yaml():
    with open("config.yaml") as file:
        return yaml.safe_load(file)

# --------------- Code for loading tweets ------------------

def get_bearer_token(config_yaml):
    return config_yaml["search_tweets_api"]["bearer_token"]

def twitter_auth_and_connect(bearer_token, url):
    print("Issuing GET request to {}".format(url))
    headers = {"Authorization": "Bearer {}".format(bearer_token)}
    response = requests.request("GET", url, headers=headers)
    return response.json()

def create_twitter_url(handle):
    max_results = 100
    # Get only tweets from last 24 hours
    yesterday = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(1)).strftime('%Y-%m-%dT%H:%M:%SZ')
    print(yesterday)

    # Create URL
    mrf = "max_results={}".format(max_results)
    q = "query=from:{}".format(handle)
    start_time = "start_time={}".format(yesterday)
    tweet_fields = "tweet.fields=text,author_id,created_at,referenced_tweets,entities"
    user_fields = "user.fields=name"
    expansions = "expansions=entities.mentions.username"
    url = "https://api.twitter.com/2/tweets/search/recent?{}&{}&{}&{}&{}&{}".format(
        mrf, q, start_time, tweet_fields, user_fields, expansions
    )
    return url

def main():
    twitter_profile = "DiscoverXnft"
    print("Loading tweets from twitter")
    url = create_twitter_url(twitter_profile)
    config_yaml = process_yaml()
    bearer_token = get_bearer_token(config_yaml)
    res_json = twitter_auth_and_connect(bearer_token, url)
    tweets = res_json["data"]
    print(tweets)

    for tweet in tweets:
        print(tweet["text"].strip("&amp;") + "\n\n")

if __name__ == "__main__":
    main()