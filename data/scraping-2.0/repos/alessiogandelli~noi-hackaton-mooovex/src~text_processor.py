#%%
import pandas as pd 
from utils import get_random_string
from dotenv import load_dotenv
import os
import langchain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from openai import OpenAI
import json 
import requests
import datetime
import langid
import subprocess

load_dotenv()

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

base_city = 'ChIJSXCeQSucgkcRKkOLNE9pK2U'

async def listen_audio(context, update):
    file = await context.bot.get_file(update.message.voice.file_id)

    print("file_id: " + str(update.message.voice.file_id))
    #save file 
    with open('data/taxi.ogg', 'wb') as f:
        await file.download_to_memory(f)
    
    #convert file
    subprocess.call([convert_script, input_file])

# transcript the audio 
def speech_to_text():
    
    client = OpenAI()
    audio_file= open("//Users/alessiogandelli/dev/cantiere/noi-hackaton-mooovex/data/taxi.mp3", "rb")
    transcript = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file
    )
    return transcript.text

# create a mp3 file from a text
def text_to_speech(text):
    client = OpenAI()

    speech_file_path = "data/reply.mp3"
    response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=text
    )
    response.stream_to_file(speech_file_path)

# parse the text to extract the fields
def parse_trip(transcript):
    prompt = PromptTemplate.from_template("""you are a voice assistant of a taxi driver, you have to extract from his query the following fields, the starting point should be or a address or a point of interest (include the city in the address), if it is a point of interest just say the name and the place without conjunctions, if no date is provided write None, if no time is provided write None, infer the language that can be it, en or de: starting_point, end_point, number_of_passengers(int), date(format it like this "%Y-%m-%d"), time(format it like this"%H:%M:%S"), language(en, de, it) .Format it as a JSON. The query is  {query}?""")
    p = prompt.format(query=transcript)
    reply = llm.invoke(p)
    trip = json.loads(reply.content)

    if trip['date'] == 'None' or trip['date'] == None: 
        trip['date'] = datetime.datetime.now().strftime("%Y-%m-%d")
    
    if trip['time'] == 'None' or trip['time'] == None:
        trip['time'] = datetime.datetime.now().strftime("%H:%M:%S")

    if trip['language'] == 'None':
        langid.set_languages(['en', 'it', 'de'])  # limit detection to these languages
        language, _ = langid.classify(transcript)
        trip['language'] = language



    return trip

# parse the answer of the users and return or yes or no
def confirm_trip(transcript):
    prompt = PromptTemplate.from_template("the user have been asked if something is correct,< {query}> is the reply, you have to tell me if the user is confirming, you can only reply <yes> or <no>, lower case, without punctuation. The user could talk in italian or english or german")
    p = prompt.format(query=transcript)
    reply = llm.invoke(p)
    print(reply.content)
    # maybe return a boolean and interpret it here 
    return reply.content

# return the number of passengers in the voice message and return it 
def number_of_passangers(transcript):

    prompt = PromptTemplate.from_template("how many passengers? reply with json format with field named 'passengers' type int: {query}")
    p = prompt.format(query=transcript)
    reply = llm.invoke(p)
    n = json.loads(reply.content)['passengers']
    print(n)
    return n

# get google place id from mooovex api
def get_place_id(trip, context, update):

    url =  'https://dev.api.mooovex.com/hackathon/autocomplete'

    data_autocomplete_start = {
        'query': trip['starting_point'],
        'language': trip['language']
    }

    data_autocomplete_end = {
        'query': trip['end_point'],
        'language': trip['language']
    }

    print(trip)


    try:
        start_response = requests.post(url, json = data_autocomplete_start, timeout=30)
        place_id_start = start_response.json()[0]['google_place_id']

    except Exception as e:
        print("did not understand the starting point\n")
        # wait for user message 
        place_id_start = None

    try:
        end_response = requests.post(url, json = data_autocomplete_end)
        place_id_end = end_response.json()[0]['google_place_id']
    except Exception as e:
        print("did not understand the destination \n", e)        
        place_id_end = None

    return place_id_start, place_id_end

# search the route in mooovex api
def search_route(place_id_start, place_id_end, trip):
    url_route = 'https://dev.api.mooovex.com/hackathon/routedetails'

    data_route = {
        'origin_google_place_id': str(place_id_start),
        'destination_google_place_id': str(place_id_end),
        'passenger_count': trip['number_of_passengers'],
        'when':{
            'date': trip['date'],
            'time':  trip['time']
        },
        'language': trip['language']
    }

    route_response = requests.post(url_route, json = data_route)

    return route_response.json()

# generate the reply that the bot should say
def generate_reply(route, trip):
    # generate the reply
    try:
        msg = 'start: '+route['origin_place']['formatted_address'] + '\n'
        msg += 'end: '+route['destination_place']['formatted_address'] + '\n'
        msg += 'number of passengers: '+str(trip['number_of_passengers']) + '\n'
        msg += 'date: '+str(trip['date']) + '\n'
        msg += 'price: '+str(route['price']) + '\n'

    except:
        msg = 'error, try again'
    

    prompt = PromptTemplate.from_template("you are the taxidriver assistant, summarize the following trip in a short and syntetic message and ask to confirm, the trip, write it in the following language{language}: {query}")
    p = prompt.format(query=msg, language=trip['language'])
    reply = llm.invoke(p)
    print(reply.content)
    return reply.content




# %%
