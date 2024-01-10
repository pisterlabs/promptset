import openai
from dotenv import load_dotenv, find_dotenv
import os
import pandas as pd 

load_dotenv(find_dotenv('./cred/.env'))

OPENAI_KEY = os.environ.get('OPENAI_KEY')

openai.api_key = OPENAI_KEY



def getFakeResponse(input_text, lang):

    babbage = 'text-babbage-001'
    curie = 'text-curie-001'
    davinci = 'text-davinci-003'

    gpt_prompt = "Generate a professional response to the below query in %s : \n\n"%(lang)


    gpt_prompt += input_text

    response = openai.Completion.create(
        engine=davinci,
        prompt=gpt_prompt,
        temperature=0.5,
        max_tokens=512,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
        )


    output_text = response['choices'][0]['text']

    return output_text


