import os
import tweepy
import openai
from pytz import timezone
from datetime import datetime, timedelta

class TweepyClient:
    def __init__(self,consumer_key,consumer_secret,access_token,access_token_secret):
        # auth = tweepy.OAuth2BearerHandler(BEARER_TOKEN)
        # self.api = tweepy.API(auth)
        # self.client = tweepy.Client(bearer_token=BEARER_TOKEN)
        self.client = tweepy.Client(
            consumer_key=consumer_key, consumer_secret=consumer_secret,
            access_token=access_token, access_token_secret=access_token_secret
        )
        
    def make_a_tweet(self,msg):
        # response = self.client.get_me()
        # print(response)
        response = self.client.create_tweet(text=msg)
        print(f"https://twitter.com/user/status/{response.data['id']}")
        
        
class OpenaiClient:
    def __init__(self,openai_key):
        openai.organization = "org-Ap7v6umG9OoBxoHFmUONp6ie"
        openai.api_key = openai_key
        # print( openai.Engine.list() )
        

    def QnA(self):
        
        today = datetime.now(timezone("America/Los_Angeles")).strftime('%B %d %Y')
        tomorrow = (datetime.now(timezone("America/Los_Angeles")) + timedelta(1)).strftime('%B %d %Y')
        
        response = openai.Completion.create(
            engine="text-davinci-001",
            prompt=f"""
            Q: Who is Batman?
            A: Batman is a fictional comic book character.
            
            Q: What is torsalplexity?
            A: ?
            
            Q: What is Devz9?
            A: ?
            
            Q: Who is George Lucas?\nA: George Lucas is American film director and producer famous for creating Star Wars.
            
            Q: What is the capital of California?
            A: Sacramento.
            
            Q: What orbits the Earth?
            A: The Moon.
            
            Q: Who is Fred Rickerson?
            A: ?
            
            Q: Any major events happening in November to December 2022?
            A: The 2022 FIFA World Cup is scheduled to be the 22nd running of the FIFA World Cup competition, the quadrennial international men's football championship contested by the national teams of the member associations of FIFA. It is scheduled to take place in Qatar from 21 November to 18 December 2022.
            
            Q: Whats happening on febuary 14th 2022?
            A: Valentine's Day is celebrated on February 14, and we are ready to shower our significant others with love and tokens of our affection. Unlike National Boyfriend Day, this day isn't just for the boyfriends — anyone and everyone can be shown some love today. 
            
            Q: What happened on september 11 2001?
            A: The September 11 attacks, also commonly referred to as 9/11, were a series of four coordinated terrorist attacks by the militant Islamist terrorist group al-Qaeda against the United States on Tuesday, September 11, 2001.
            
            Q: How many moons does Mars have?
            A: Two, Phobos and Deimos.
            
            Q:Today is {today} and tomrrow is {tomorrow}. Tell me what is happening tomorrow? Any predictions for cryptocurrency?
            A:""",
            temperature=0,
            max_tokens=60,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
            )
        print(response)
        return response['choices'][0]['text']
