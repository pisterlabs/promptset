from twttr_functions import twttr_handler
from openAI import create_new_tweet, create_tweet_response
import random
import json
import time




def post_tweet():
    max_retries = 5
    for i in range(max_retries):
        try:
            print(post_tweet_new())
            break  # if the function runs successfully, break the loop
        except Exception as e:
            if i < max_retries - 1:  # i starts at 0, so we subtract 1
                print('Error encountered while posting a new tweet: ', str(e))
                print(f'Retrying after 5 seconds... (Attempt {i+2}/{max_retries})', flush=True)  # i+2 because i starts at 0
                time.sleep(5)
            else:
                print('Error encountered on the final attempt. No further retries will be made.')
                raise  # re-raise the last exception

def post_tweet_new():
    # Choose between creating a new tweet or responding to a tweet
    new_tweet_probability = 0.0  # Probability of writing a new tweet
    rand_num = random.random()  # Generate a random number between 0 and 1

    try:
        if rand_num < new_tweet_probability:
                # Load topics from json file
            with open('src/new_engagement/topics.json', 'r') as f:
                topics = json.load(f)['topics']
            # This branch will be taken 5% of the time
            filtered_list = [topic for topic in topics if not topic['additional_context']]
            tweet = create_new_tweet.generate_tweet(filtered_list)
            print('Tweet: ', tweet)
            print('Posted new tweet')
            print (twttr_handler.decide_action('tweet', tweet=tweet), flush=True)
        else:
            # This branch will be taken 95% of the time
            tweet_id = twttr_handler.decide_action('random-timeline')
            print('Tweet ID: ', tweet_id)
            tweet_conversation = twttr_handler.decide_action('conversation', tweet_id=tweet_id)
            print('Tweet conversation: ', tweet_conversation)
            tweet = create_tweet_response.build_message(tweet_conversation)
            print('Tweet: ', tweet)
            print(twttr_handler.decide_action('tweet', tweet=tweet, tweet_id=tweet_id), flush=True)
            print('Replied to tweet')
    except Exception as e:
        raise e

if __name__ == '__main__':
    post_tweet()