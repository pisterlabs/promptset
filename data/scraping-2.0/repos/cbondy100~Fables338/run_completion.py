# check out this for more info
# https://beta.openai.com/pipdocs/guides/fine-tuning
# particularly, there's a command on there that shows you how to set your api key in the command line so that this script works

import os
import openai

class openai_engine:
    @classmethod
    def run_completion(cls, api_key, input):
        openai.api_key = api_key
        response = openai.Completion.create(
        model="davinci:ft-personal-2022-12-06-04-03-15",
        prompt=input,
        max_tokens=1000,
        temperature=0.9,
        #top_p=0.3,
        stop="THEEND"
        )
        output = response["choices"][0]["text"]
        output = output[22:].strip()
        #print(output)
        return output

    @classmethod
    def generate_image(cls, api_key, prompt):
        response = openai.Image.create(
        prompt=prompt,
        n=1,
        size="1024x1024"
        )
        output = response['data'][0]['url']
        #print(output)
        return output