from openai import OpenAI
from django.http import HttpResponse
import os

api_key_env = os.environ.get('OPENAI_API_KEY')

def SongInput(input):
    client = OpenAI(
    api_key=api_key_env
    )

    completion = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "system", "content": "You are a Song assistant, skilled in giving songs related to the one provided, providing songs from different artists.Will respond with just the songs with nothing but a newline in between songs."},
        {"role": "user", "content": f"From this song:{input}, give me a list of 5 songs that are like this song wether in genre, style or artist. Parameters of response should just be the songs name with no artist and nothing else with just a newline in between and no commas, numbers,quotes,and spaces in between the songs"}
    ]
        )
    
    print(api_key_env)
    return(completion.choices[0].message)
    
    
