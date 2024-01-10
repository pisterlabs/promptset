import openai
import json
import os
import random
import requests

import codecs
from boto3 import Session
from boto3 import resource
from boto3 import client

session = Session(region_name="us-east-1")
polly = session.client("polly")

s3 = resource('s3')
bucket_name = 'gpt3-video-scripts'
bucket = s3.Bucket(bucket_name)
s3_client = client('s3')
lambda_client = client('lambda')

def invoke_lambda(function_name, payload):
    response = lambda_client.invoke(
        FunctionName=function_name,
        InvocationType='Event',
        Payload=json.dumps(payload)
    )
    return None

def process_audio(directory_name, script_arr, neural=False):
    voiceIDs = [
        "Matthew",
        "Russell",
        "Nicole",
        "Emma",
        "Ivy",
        "Kendra",
        "Kimberly",
        "Joey",
        "Justin"
    ]
    neural_voices = [
      "Danielle",
      "Gregory",
      "Kevin",
      "Joanna",
      "Matthew",
      "Ruth",
      "Stephen",
      "Ayanda",
      "Niamh",
      "Aria", 
      "Arthur",
      "Olivia"
    ]
    if neural:
      # join the two lists
      voiceIDs.extend(neural_voices)
    voices = {}
    line_num = 0

    for voice_obj in script_arr:
        voice, line = voice_obj
        if voice not in voices:
            voiceID = random.choice(voiceIDs)
            voices[voice] = voiceID
            voiceIDs.remove(voiceID)

        # now process the audio
        filename = f"{directory_name}/audio/{line_num}.mp3"
        try:
            tts(line, voices[voice], filename)
        except e as Exception:
            print(f"{voices[voice]} is a failure")
            print("Exception: ", e)
        line_num += 1


def tts(text, voiceId, filename):
    response = polly.synthesize_speech(
        Text=text,
        OutputFormat="mp3",
        Engine="neural",
        VoiceId=voiceId
    )
    stream = response["AudioStream"]
    bucket.put_object(Key=filename, Body=stream.read())


def lambda_handler(event, context):
    folder_name = event['folder_name']
    script_array = event['script_array']
    caption_enabled = event['caption_enabled']
    print("Caption enabled: ", caption_enabled)
    
    # audio
    process_audio(folder_name, script_array, True)
    
    # invoke lambda to process audio
    invoke_lambda("combine_audio", {"folder_name": folder_name, "caption_enabled": caption_enabled})

    return {
        'statusCode': 200,
        'body': json.dumps({"folder_name": folder_name})
    }
