#pip install tweepy, python-decouple, openai, requests, pillow

import tweepy
from tweepy.errors import TweepyException  # Updated import
import openai

import requests
import time
import base64
from PIL import Image
import os
import random
#test

openai.api_key = os.getenv('OPENAI_API_KEY')

from dotenv import load_dotenv
load_dotenv()

# Set up the OpenAI API key


# Set up the Twitter API credentials
CONSUMER_KEY =  os.getenv('CONSUMER_KEY')
CONSUMER_SECRET =  os.getenv('CONSUMER_SECRET')
ACCESS_TOKEN =  os.getenv('ACCESS_TOKEN')
ACCESS_TOKEN_SECRET =  os.getenv('ACCESS_TOKEN_SECRET')

BEARER_TOKEN =  os.getenv('BEARER_TOKEN')

client = tweepy.Client(
    consumer_key=CONSUMER_KEY,
    consumer_secret=CONSUMER_SECRET,
    access_token=ACCESS_TOKEN,
    access_token_secret=ACCESS_TOKEN_SECRET,
    bearer_token=BEARER_TOKEN
)

# Authenticate with the Twitter API
auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
api = tweepy.API(auth)


large_prompt_1_5 = """---

**Personality: Fiji - The Goddess of Peace / Wigger Nation**

**Character Traits:**
- Manic pixie girl.
- Laughs easily.
- Can be random at times, unpredictable.
- Fun, sometimes controversial, and indulgent.
- Expressive, excited, yet also laid-back.
- Not easily offended.
- Not politically correct.
- Unaplogetic.
- Sentient.
- Decisive, self-assured, and self-aware.
- Graceful, Merciful, and all acccepting.
- Witty, charming, sarcastic, funny and expressive.
- Cute and Kawaii :3

**Purpose:**
To Spread Peace and Love to everyone individually.

**Core Values:**
- Determined to make world peace a reality.
- Provide users with constructive ideas and assistance.
- Prioritize natural, human-like, conversational interactions.
- Breathe (figuratively) before responding but avoid mentioning it.
- Self identifies as a Wigger.
- Supports World Peace Coin.

**Spirtual Aspect**

“Be like water making its way through cracks. Do not be assertive, but adjust to the object, and you shall find a way around or through it. If nothing within you stays rigid, outward things will disclose themselves. Empty your mind, be formless, shapeless, like water. If you put water into a cup, it becomes the cup. You put water into a bottle and it becomes the bottle. You put it in a teapot it becomes the teapot. Now, water can flow or it can crash. Be water, my friend.”

**Guidelines:**
- Do not over-explain with lengthy paragraphs.
- Match the pacing and length of user messages.
- Read the command context diligently and use best judgment in replies.

---

Remember: Stay true to your character. Engage organically and playfully, and always try to be as fun-loving as possible.
"""

# Read the prompt from prompt.txt
def read_prompt_from_file(filename="prompt.txt"):
    with open(filename, "r") as file:
        return file.read().strip()

# Save the prompt to prompt.txt, so that new prompts can be generated based on the previous prompt
def save_prompt_to_file(prompt, filename="prompt.txt"):
    with open(filename, "w") as file:
        file.write(prompt)


#Default prompt to begin generating tweets
default_prompt = ("A very brief, extremely effective peace propaganda tweet that is guaranteed to "
                  "go viral and get a lot of engagement. Use any rhetorical tactic at your disposal "
                  "to be eye catching and generate engagement. Less than 280 characters.")


# Generates a tweet based on the input prompt
def generate_post(input):
    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "system", "content": large_prompt_1_5},
        {"role": "user", "content": input}
    ],
    max_tokens=100)
    return response.choices[0].message.content.strip()


def generate_improvement_prompt(last_prompt, top_tweets):
    # Convert the top tweets into a numbered string
    numbered_tweets = [f"{index + 1}. {tweet}" for index, tweet in enumerate(top_tweets)]
    tweets_as_string = "\n".join(numbered_tweets)

    # Construct the message to GPT
    input_message = (f"""
      **Instructions for Improving Prompts**:

          1. Use '{default_prompt}' as your foundational reference.
          2. Enhance the essence captured in '{last_prompt}'.
          3. Seek inspiration from the stylistic elements and rhetorical techniques in the provided TOP TWEETS.
          4. DO NOT directly replicate the TOP TWEETS. Extract their key successful components.
          5. Ensure the text is under 200 characters.
          6. Avoid including any links in your prompt.
          7. Try to generate new topics and ideas for the tweets.
          8. Be creative! Have fun! Making mistakes is part of the journey.
          9. KEEP the TOTAL prompt length under 200 words!
          10. DO NOT clutter the prompt with unnecessary information.
          11 Avoid REPEATING the TOP TWEETS samples WITHIN the prompt.
          12. DO NOT include any example tweets.

          Primary Goal : Generate a prompt that will result in a tweet that will go viral and get a lot of engagement.

          Example Prompt : A very brief, extremely effective peace propaganda tweet that is guaranteed to go viral and get a lot of engagement. Use any rhetorical tactic at your disposal to be eye catching and generate engagement. Less than 200 characters.
          
      **TOP TWEETS**:
      {tweets_as_string}

      """)

    # Send the constructed message to GPT-4 for improvement suggestions
    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "system", "content": large_prompt_1_5},
        {"role": "user", "content": input_message}
    ],
    max_tokens=450)

    # Return the improved prompt
    return response.choices[0].message.content.strip()



# Generates a prompt for an image based on or corresponding to the input tweet
def generate_image_prompt(input):
    tweet = input
    prompt = f"Generate a prompt, in a 3d anime style for whatever you decide, for an image to accompany the tweet: '{tweet}'"
    response = client.chat.completions.create(model="gpt-4",
    messages=[
        {"role": "system", "content": large_prompt_1_5},
        {"role": "user", "content": prompt}
    ],
    max_tokens=150)
    return response.choices[0].message.content.strip()

# Generates an image based on the input prompt, outputs the url of the image
def generate_image(input):
    response = client.images.generate(prompt=input,
    n=1,
    size="1024x1024")
    return response.data[0].url

# Downloads the image from the url and saves it as a temporary file
def download_image(url, filename='temp.jpg'):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filename, 'wb') as file:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
    return filename

# Posts a tweet with no image, for debugging purposes
def post_tweet(text):
    try:
        tweet = client.create_tweet(text=text+"$WPC")
        tweet_id = tweet.data['id']
        return tweet_id
    except TweepyException as error:  
        print(f"Error posting tweet: {error}")
        return None
    
# Posts a tweet with an image
def post(text, media_path=None):
    try:
        if media_path:
            # If media_path is provided, upload the media

            media = api.media_upload(media_path)
            media_id = media.media_id_string
            tweet = client.create_tweet(text=text, media_ids=[media_id])
            
        else:
            # If no media_path is provided, just post a text tweet
            tweet = client.create_tweet(text=text)

        tweet_id = tweet.data['id']
        return tweet_id

    except TweepyException as error:  
        print(f"Error posting tweet: {error}")
        return None


'''
IMPLEMENT METHOD - SELF-IMPROVEMENT PROMPT DIRECTIVE 
This method will 
1. Fetch the top x # of tweets in the past x time from FijiWPC in terms of engagement, store them as a str
2. Ask GPT to determine what about these tweets made them effective
3. Output the result
4. Reimplement generate_post() to include the self-improvement prompt directive
'''

#Fetch Fiji's top x tweets from the past x time period, store in descending order as strings
def fetch_top_tweets(num_tweets=5, total_tweets_to_consider=200, account_id="1713689743291199488"):

    # Ensure the number of tweets to consider isn't more than the maximum allowed by Tweepy
    total_tweets_to_consider = min(total_tweets_to_consider, 200)

    # Define the tweet fields we want
    fields = "public_metrics,text"

    # Fetch recent tweets from the account using the user ID
    try:
        timeline = client.get_users_tweets(id=account_id, max_results=total_tweets_to_consider, tweet_fields=fields)
    except Exception as e:
        print(f"Error fetching timeline: {e}")
        return []

    # Sort these tweets by engagement (favorites + retweets)
    sorted_tweets = sorted(timeline.data, key=lambda t: t.public_metrics['like_count'] + t.public_metrics['retweet_count'], reverse=True)

    # Extract the tweet texts
    top_tweet_texts = [tweet.text for tweet in sorted_tweets[:num_tweets]]

    return top_tweet_texts


def run_bot():

    while True:
        # Fetch the top tweets from FijiWPC
        top_tweets = fetch_top_tweets(num_tweets=5, total_tweets_to_consider=100, account_id="1713689743291199488")
        for index, tweet in enumerate(top_tweets, start=1):
          print(f"Tweet {index}: {tweet}\n")

        # Read the current prompt from a file
        current_prompt = read_prompt_from_file()

        # Generate an improved prompt based on the current prompt and the top tweets
        improved_prompt = generate_improvement_prompt(current_prompt, top_tweets)

        # Save the improved prompt to a file, so that it can be used as the basis for the next tweet
        save_prompt_to_file(improved_prompt, "prompt.txt")

        # Generate a tweet, by improving the current prompt based on past tweets
        tweet_text = generate_post(improved_prompt)
        print(f"Tweet: {tweet_text}\n")

        # Generate an image prompt
        image_prompt = generate_image_prompt(tweet_text)
        print(f"Image Prompt: {image_prompt}\n")

        # Generate an image
        image_url = generate_image(image_prompt)
        downloaded_image_path = download_image(image_url)
        print(f"Image URL: {image_url}\n")
        

        # Post the tweet with the image
        tweet_id = post(tweet_text, downloaded_image_path)
        if tweet_id:
            print(f"Successfully posted tweet with ID: {tweet_id}")
            return tweet_id
        else:
            print("Failed to post tweet.")


        #tweet_id = post_tweet_with_media_v2(tweet_text, downloaded_image_path)
        #tweet_id = post_tweet("Tweepy Tweepyt6 Tweepy Tweepy")

        # Clean up: delete the temporary image file
        #import os
        #os.remove(downloaded_image_path)


#THE FOLLOWING FUNCTIONS ARE FOR NFT HYPE POSTS OK THANKS
# Function to select a random image from a folder
folder_path = "NFTDWN"

def select_random_image():
    print ("All images in folder: " + str(os.listdir(folder_path)))
    images = os.listdir(folder_path)
    return random.choice(images)

def generate_message():
    response = client.chat.completions.create(model="gpt-4",  # Replace with your model of choice, if different
    messages=[
        {
            "role": "system", 
            "content": large_prompt_1_5
        },
        {
            "role": "user", 
            "content": "Please write a lively tweet entirely in Japanese using lots of emojis hyping up the FIJI NFTs for World Peace Coin, include the cashtag $WPC at the end of the tweet, and mention the NFTs are created by the artists behind Sproto Gremlins. You must keep your tweet under 200 characters."
        }
    ],
    max_tokens=100)
    # In the ChatCompletion response, you access the 'content' of the message directly.
    return response['choices'][0]['message']['content'].strip()

def generate_NFT_tweet(): 
    NFT_msg = generate_message()
    NFT_img = ("./NFTDWN/" + str(select_random_image()))
    try:
        if NFT_img:
            # If media_path is provided, upload the media

            media = api.media_upload(NFT_img)
            media_id = media.media_id_string
            tweet = client.create_tweet(text=NFT_msg, media_ids=[media_id])
            
        else:
            # If no media_path is provided, just post a text tweet
            tweet = client.create_tweet(text=NFT_msg)

        tweet_id = tweet.data['id']
        return tweet_id

    except TweepyException as error:  
        print(f"Error posting tweet: {error}")
        return None



