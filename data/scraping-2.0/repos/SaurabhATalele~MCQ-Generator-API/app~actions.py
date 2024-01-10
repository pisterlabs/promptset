import os
import openai
# openai.api_key = os.getenv("OPENAI_API_KEY")
from decouple import config

OPENAI_API_KEY = config('OPENAI_API_KEY')
print(OPENAI_API_KEY)
openai.api_key = OPENAI_API_KEY
def get_gpt3_response(prompt):
    # return openai.ChatCompletion.create(
    # model="gpt-3.5-turbo",
   
    #  messages=[
      
    #     {"role": "user", "content": "who won the cricket World cup in 2019."},
       
    # ],
    # max_tokens=500,
    # temperature=0
    # )['choices'][0]['text']
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt},
        
    ]
    )
    # print(response)
    return response
