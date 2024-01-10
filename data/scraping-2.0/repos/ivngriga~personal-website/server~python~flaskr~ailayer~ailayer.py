import os
import openai

class AIModel():
    def __init__(self,apikey):
        openai.api_key = apikey
        

    def complete(self,prompt, maxtokens,temp=1.0,freq_pen=0.6):
        completion=openai.Completion.create(
            model="text-davinci-003",
            prompt=prompt,
            max_tokens=maxtokens,
            temperature=temp,
            frequency_penalty=freq_pen
        )
        print(completion)
        return completion
