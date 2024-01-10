import openai
from recommender import config
from recommender import templates
import random
import csv
from easydict import EasyDict as edict
event_types = ['sports', 'music']
openai.api_key = config.OPENAI_API_KEY

class Assistant(object):
    def __init__(self, engine="text-davinci-002"):
        print("Initializing Assistant...")
        self.engine = engine
        self.is_initialized = False
        
        
    
    def initialize(self, user):
        if(not self.is_initialized):
            self.is_initialized = True
            self.user = user
            self.user_story = templates.my_story.format(self.user.username, self.user.birth_year, 
                                                            self.user.birth_place, self.user.current_place, 
                                                            self.user.favorite_band, self.user.favorite_film)
        else:
            print("Assistant was already initialized")
        print("Assistant Ready...")

    def add_info_user_story(self,info):
        self.user_story += (info+". ")

    def recommend_film(self):
        recommendation = self.send_query(templates.film_query.format(self.user.favorite_film))
        recommendation = recommendation.replace('\n\n','').split('\n')
        recommendation = recommendation[random.randint(1,3)][3:]
        return (recommendation,
                self.send_query(templates.query.format("the film" + recommendation), max_tokens=256, temperature=0.4))

    def recommend_band(self):
        recommendation = self.send_query(templates.band_query.format(self.user.favorite_band))
        print(recommendation)
        recommendation = recommendation.replace('\n\n','').split('\n')
        print(recommendation)

        recommendation = recommendation[random.randint(0,2)][3:]
        print(recommendation)

        return (recommendation, 
                self.send_query(templates.query.format("the artist " + recommendation), max_tokens=256, temperature=0.4))

    def recommend_song(self):
        recommendation = self.send_query(templates.song_query.format(self.user.favorite_band))
        recommendation = recommendation.replace('\n\n','').split('\n')
        recommendation = recommendation[random.randint(0,2)][3:]
        return recommendation

    def recommend_event(self):
        year = int(self.user.birth_year)+random.randint(15,50)
        year = int((year/10)*10)
        print("year: {}".format(year))
        recommendation = self.send_query(templates.historical_query.format(event_types[random.randint(0,1)], self.user.birth_place, year)).split('.')[0]
        return (recommendation, self.send_query(templates.query.format(recommendation),max_tokens=256, temperature=0.6))
    
    def ask(self, question):
        return self.send_query(self.user_story + "\n\nHuman: " + question + "\n\nAssistant")
    
    def send_query(self,msg,max_tokens=32, temperature = 0.4):
        response = openai.Completion.create(
        engine=self.engine,
        prompt=msg,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.8,
        frequency_penalty=0,
        presence_penalty=0
        )
        return response["choices"][0]["text"]
