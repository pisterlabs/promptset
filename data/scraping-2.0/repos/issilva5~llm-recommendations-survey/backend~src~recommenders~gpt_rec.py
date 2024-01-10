import openai
import os
import json

from ..utils import get_movie_poster

class GPTRecommender():

    def __init__(self, model = "gpt-3.5-turbo") -> None:
        openai.api_key = os.getenv("OPEN_API_KEY")
        self.model = model

    def get_recommendations(self, prompt):

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that recommends movies to a user based on his previous watched movies."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = openai.ChatCompletion.create(model=self.model, messages=messages,
                                            temperature=0, top_p=1, frequency_penalty=0,
                                            presence_penalty=0, n=1)
        
        recommendations = json.loads(response["choices"][0]["message"]["content"])
        for movie in recommendations['recommendations']:
            movie['poster'] = get_movie_poster(movie['imdbID'])
            movie['recommender'] = self.model
        
        return recommendations['recommendations']
