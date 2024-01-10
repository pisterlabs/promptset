import re
import asyncio
import os
import openai



class BravoSis:
    def __init__(self):
        self.openai_api_key = "" # set your own api value
        self.openai_api_key = "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" #get this value from https://beta.openai.com/.
        self.model = "text-davinci-003" # use any of these [text-davinci-002,text-davinci-001]
        self.mxtoken = 1080 #can decrese/increse with reaspect to result previlage


    def ai(self,query):
        openai.api_key = self.openai_api_key
        completion = openai.Completion.create(engine=self.model, prompt=query, max_tokens=self.mxtoken, n=1, stop=None,temperature=0.7)
        result = completion.choices[0].text
        return result


