from operator import length_hint
import os
from urllib import response
import openai
import config
openai.api_key = config.OPENAI_API_KEY

def ai_program(query):
    
    response = openai.Completion.create(
        model="code-davinci-002",
        prompt="\"\"\"\n code in python:\n {} \n\"\"\"\n".format(query),
        temperature=0.00,
        max_tokens=250,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0)

    if 'choices' in response:
        x = response['choices']
        if len(x) > 0:
            answer = x[0]['text']
        else:
            answer = "No answer found"
    else:
        answer = "No answer found"

    return answer


