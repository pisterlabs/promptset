import openai
import json
import os
from dotenv import load_dotenv
load_dotenv('.env')
QA_PROMPT = """Based on the chat history given below
{chat_summary}
You need to fill in the JSON object given below,Keep a key empty if nothing can be extracted for a particular key:-
{{
    "date_published":{{
        "greater_than":"",
        "less_than":""
    }},
    "duration":{{
        "greater_than":"",
        "less_than":""
    }},
    "Genres":[],
    "actors":[],
    "keywords":[],
    "Director":[],
    "chat_summary":"",
    "RatingValue":{{
        "greater_than":"",
        "less_than":""
    }},
"answer":"",
}}
Information of data types that go into the keys is given below:-
date_published:- Date at which movie was published (Date format YYYY-MM-DD, Consider today date as 20 october 2023)
duration:- Number of minutes that the user wants the movie to last, Just an integer value respresenting number
Genres:- List of Genres that movie can be
actors:- List of actors of the movie
keywords:- List of keywords that movie can be
Director:- Director of the movie, If any mentioned in chat
chat_summary:- Summary of the chat above
RatingValue:- IMDB Rating of the movie
answer:- A string representing how an AI would reply with fun message considering it has provided a lot of movie results, Don't take any movie names in this answer
Return the Json Response only after filling the extracted values, And don't add any comments

JSON Respone:"""

PLAYLIST_PROMPT="""
Based  on the titles given, Please suggest a name for the playlist:
{titles}

Return response in json with "name" as the key"""
openai.api_key = os.environ.get("OPENAI_API_KEY")
def get_response(chat_history):
    print(chat_history)
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    temperature=0,
    messages=[
        {"role": "user", "content": QA_PROMPT.format(chat_summary=str(chat_history))}
    ]
    )
    print(completion.choices[0].message)
    return json.loads(completion.choices[0].message.content)

def get_playlist_name(titles):
    completion = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": PLAYLIST_PROMPT.format(titles=str(titles))}
    ]
    )
    return json.loads(completion.choices[0].message.content)
