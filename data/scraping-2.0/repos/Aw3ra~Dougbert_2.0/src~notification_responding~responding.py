import json
from commands import decide_command
from twttr_functions import twttr_handler
from openAI import create_tweet_response
from data_ingestion import request_info
import re
import time
from dotenv import load_dotenv
from pymongo import MongoClient
import os
import random
from notification_responding import doug_logging

def perform_action(action, tweet_conversation, tweet_id = None):
    try:
        # If the last tweet in the conversation is a message by the bot, ignore it
        if not tweet_conversation[-1]['content'].lower().startswith(os.getenv("TWITTER_HANDLE").lower()):
            if action == 'B':
                # Create ai response first
                tweet_response = create_tweet_response.build_message(tweet_conversation)
                twttr_handler.decide_action('tweet', tweet = tweet_response, tweet_id = tweet_id)
            elif action == 'C':
                twttr_handler.decide_action('retweet', tweet_id = tweet_id)
            elif action == 'D':
                twttr_handler.decide_action('like', tweet_id = tweet_id)
            elif action == 'E':
                print('Follow user')
            elif action == 'F':
                # Create ai response first
                tweet_response = create_tweet_response.build_message(tweet_conversation, prior_known_info = request_info.return_system_message(tweet_conversation))
                twttr_handler.decide_action('tweet', tweet = tweet_response, tweet_id = tweet_id)
            elif action == 'G':
                print('Check Solana price')
            elif action == 'H':
                print('Lookup user profile')
            elif action == 'I':
                print('Sending tip')
                # Create message along with the tip
                tweet_response = create_tweet_response.build_tip_message(tweet_conversation)
                # To user name is the last user name in the thread
                return twttr_handler.decide_action('send-dm', tweet = tweet_response, to_user_name = tweet_conversation[-1]['content'].split(':')[0])
        else:
            print('Last tweet in conversation is by bot, ignoring')
    except Exception as e:
        raise e

def respond_to_notification():
    load_dotenv()
    client = MongoClient(os.getenv("DATABASE_URL"))
    with open('src/profile.json', 'r', encoding='utf-8') as f:
        records = json.load(f)['config_data']
        database = records["Database_records"]['Database']
        collection = records["Database_records"]["Collection"]
        logs = records["Database_records"]["Logs"]
        logs_collection = client[database][logs]
        notifications_collection = client[database][collection]

    # Function to update actioned to true
    def update_actioned(tweet):
        notifications_collection.update_one({"tweetId": tweet}, {"$set": {"actioned": True}})

    def get_tweet():
        tweets = list(notifications_collection.find({"actioned": False}))
        if len(tweets) == 0:
            return None
        return tweets[0]
    print('Finding tweet to respond to')
    # 50/50 chance of getting a notification tweet or a random tweet
    update_database = False
    if random.randint(0,10) == 0:
        print('Getting random tweet')
        tweet = twttr_handler.decide_action('random-timeline')
    else:
        print('Getting notification tweet')
        tweet = get_tweet()
        if tweet == None:
            print('Getting random tweet')
            tweet = twttr_handler.decide_action('random-timeline')
        else:
            update_database = True
            tweet = tweet['tweetId']
    try:

        if tweet == None:
            return
        print(tweet)
        tweet_conversation = twttr_handler.decide_action('conversation', tweet_id = tweet)
        user_name = tweet_conversation[-1]['content'].split(':')[0]
        print('User name: ', user_name)
        if user_name.lower() == os.getenv("TWITTER_HANDLE").lower():
            print('Tweet is by bot, ignoring: ', tweet)
            raise Exception('Tweet is by bot, ignoring')
        if len(tweet_conversation) != 0 and tweet_conversation is not None:
            actions = decide_command.decide_command(tweet_conversation)
            actions_list = [action.strip() for action in re.search(r"\[(.*?)\]", actions).group(1).split(',')]
            if 'A' not in actions_list:
                for action in actions_list:
                    perform_action(action, tweet_conversation, tweet_id=tweet)

        doug_logging.create_log(tweet, tweet_conversation[-1]['content'], actions_list)

    except Exception as e:
        print(f'Error in respond_to_notification: {e}')
        doug_logging.create_log(tweet, tweet_conversation[-1]['content'], Exception)
    finally:
        # Create a log using the tweet id, tweet text of the last tweet in the conversation and the actions
        if update_database:
            update_actioned(tweet)

def testing():
    # tweet_conversation = twttr_handler.decide_action('conversation', tweet_id = "1664497913299632128")
    print(twttr_handler.decide_action('send-dm', tweet = 'Thankyou so much for interacting with me! Here\'s a little tip for your time!', to_user_name = '_qudo'))
    # return perform_action('I', tweet_conversation)

# twitter_handler.decide_action('reply_to_tweet', tweet = 'Test tweet', tweet_id = asyncio.run(get_tweet()).tweetId)


