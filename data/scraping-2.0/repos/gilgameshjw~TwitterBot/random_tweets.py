
import sys
import openai
import random
import json
import names
import tqdm
import yaml


# reads above file from config.yaml file
config = yaml.load(open('config.yaml'), Loader=yaml.FullLoader)
num_tweets = config['twitter']['num_tweets']
engine = config['openai']['openai_engine']
api_key = config['openai']['openai_key']

openai.api_key = api_key

# read the twitter handle from the command line
if len(sys.argv) != 3:
    print("Usage: python random_tweets.py <character> <twitter_handle>")
    sys.exit(1)
character = " ".join(sys.argv[1].split("_"))
twitter_handle = sys.argv[2]


# Define a function for generating a tweet or retweet
def generate_tweet():
    prompt = f"Can you generate a short {' retweet' if random.random() < 0.5 else ' tweet'} a character like {character} could have written?"
    response = openai.Completion.create(
        engine=engine,
        prompt=prompt,
        max_tokens=100,
        temperature=0.7,
        n=1,
        stop=None
    )
    return response.choices[0].text.strip()

# run the random tweets generator
with open('data/twitter_data_'+twitter_handle+'.jsonl', 'a') as file:
    
    for _ in tqdm.tqdm(range(num_tweets)):
        if random.random() < 0.5:  # Generate a random tweet
            tweet = generate_tweet()
            tweet_data = {
                'created_at': 'Mon Jun 02 00:00:00 +0000 2023',
                'id_str': str(random.randint(100000000000000000, 999999999999999999)),
                'text': tweet,
                'user': {'screen_name': twitter_handle}
            }
        else:  # Generate a retweet of a random tweet
            retweet_id = str(random.randint(100000000000000000, 999999999999999999))
            mock_user = names.get_full_name().replace(" ", "")
            tweet_data = {
                'created_at': 'Mon Jun 02 00:00:00 +0000 2023',
                'id_str': str(random.randint(100000000000000000, 999999999999999999)),
                'text': generate_tweet(),
                'retweeted_status': {
                    'created_at': 'Mon Jun 01 00:00:00 +0000 2023',
                    'id_str': retweet_id,
                    'text': generate_tweet(),
                    'user': {'screen_name': mock_user}
                },
                'user': {'screen_name': twitter_handle}
            }
        
        file.write(json.dumps(tweet_data)+'\n')

