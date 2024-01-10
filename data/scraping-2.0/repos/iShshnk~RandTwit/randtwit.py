from os import access
import tweepy
import openai
import random

class twitterbot():
    api_key = "T4L8qQti3pJkfsIEszzfi3bOY"
    api_secret = "Ge3EJyWG6LDoQXATni3OchiKrTHrXjF3BF34Vm5QjHlPCLuZMH"
    access_key = "1656209603129131008-TD5uClxdTO9uRulnT7Iq4luVEdfnLv"
    access_key_secret = "Z5yZcvtAEirHCWwhlmESaI83QYX425y1MxId4CPR6hTUd"
    openai_key = "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"

    auth = tweepy.OAuthHandler(api_key, api_secret)
    auth.set_access_token (access_key, access_key_secret)
    api = tweepy.API(auth, wait_on_rate_limit=True)
    openai.api_key = openai_key

    '''
    response = openai.Completion.create(engine= "text-davinci-001", prompt="tweet about killing kim jong-un", max_tokens=200)
    text = response.choices[0].text
    print(text)
    api.update_status(text) 
    '''
    prompts = [       {
                "hashtag": "#elonmusk",
                "text": "tweet something cool about elon musk aquiring twitter"
            },
            {
                "hashtag": "#quote",
                "text": "tweet a motivational quote"
            },
            {
                "hashtag": "#ai",
                "text": "tweet latest developments in artificial intelligence"
            },

            {
                "hashtag": "#startup",
                "text": "tweet about startups in India "
            }
    ]


    def __init__(self):
            error = 1
            while(error == 1):
                tweet = self.create_tweet()
                try:
                    error = 0
                    self.api.update_status(tweet)
                except:
                    error = 1
        
    def create_tweet(self):
            chosen_prompt = random.choice(self.prompts)
            text = chosen_prompt["text"]
            hashtags = chosen_prompt["hashtag"]

            response = openai.Completion.create(
                engine="text-davinci-001",
                prompt=text,
                max_tokens=200,
            )

            tweet = response.choices[0].text
            tweet = tweet + " " + hashtags
            return tweet

twitter = twitterbot()
twitter.create_tweet()
