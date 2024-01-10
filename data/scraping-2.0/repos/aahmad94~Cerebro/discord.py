import os
import openai
import time

from parse_twitter import ParseTwitter
from datetime import datetime
from discord_webhook import DiscordWebhook
from dotenv import load_dotenv

class TwitterToDiscord:

    def __init__(self, webhook_url, users, tweets):
        load_dotenv()
        openai.organization = "org-zeOV533K5hdpS1CFEc4Ph8Mh"
        openai.api_key = os.getenv("OPENAI_KEY")
        
        self.tweets = tweets
        self.webhook_url = webhook_url
        self.users = users
        self.get_user_tweets()


    def get_user_tweets(self):
        for user in self.users:
            parser = ParseTwitter(user)
            parser.initAction(parser.getLastTweetAction)
            
            tweet_url = parser.tweet_info["tweet_url"]
            tweet_text = parser.tweet_info["text"]
            content = f"{self.shorten_post(tweet_text)}"

            # only fwd tweets not in dict & only after dict is initialized w/ n items
            if tweet_text and tweet_url and not self.tweets.get(tweet_url):
                self.tweets[tweet_url] = True
                gpt_result = f"\n\n\n__ChatGPT__\n\n{self.ask_gpt(tweet_text)}"
                if len(content) < 250:
                    gpt_result = '' 
                
                if len(self.tweets) >= len(self.users):
                    self.fwd_tweet(f"```\n__{user.upper()}__\n\n{content}{gpt_result}```")
                    self.fwd_tweet(f"<{tweet_url}>")


    def shorten_post(self, text, trim_len=250):
        if len(text) > trim_len:
            return f"{text[:trim_len]}..."
        return text


    def ask_gpt(self, tweet_text):
        prompt = "Explain the following tweet (along with any acronyms if needed) in as little number of words possible. \
                  If you can't quite understand the tweet, answer with 'Nothing to summarize'. \n\n"
        messages = [{"role": "user", "content": prompt + tweet_text}]
        
        try:
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages)
            print("ChatGPT API endpoint success")
        except: 
            print("ChatGPT API endpoint failure\n")
            return ''

        return chat.choices[0].message.content
                   

    def fwd_tweet(self, content):
        print(f"CONTENT:\n {content}\n")
        DiscordWebhook(url=self.webhook_url, content=content).execute()
