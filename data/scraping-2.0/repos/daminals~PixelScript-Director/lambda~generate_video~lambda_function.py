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

def split_script(script):
    # read everything here as voice 1: "\nVoice 1:\n"
    # split the script into multiple voices with each voice indicated by a newline, a colon, voice and the number of this character currently speaking
    lines = script.split("\n")
    result = []
    current_voice = None
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("Voice"):
            current_voice = line.strip()
        elif line.startswith("Scene"):
            pass  # ignore scene
        else:
            if current_voice is not None:
                result.append([current_voice, line.strip()])
        i += 1
    return result

def generate_foldername(director):
    # generate a random 8 digit number
    num = random.randint(10000000, 99999999)
    director = director.replace(" ", "-")
    return f"{director}-{num}"

def lambda_handler(event, context):
    topic = None 
    directory_name = None
    script = None
    caption_enabled = False
    
    body = json.loads(event['body'])
    # return {
    #     'statusCode': 200,
    #     'body': json.dumps(body)
    # }
    print("body", body)
    
    if 'topic' in body:
        topic = body['topic']
    if 'directory' in body:
        directory_name = body['directory']
    if 'script' in body:
        script = body['script']
    if 'caption_enabled' in body:
        print("Body Caption enabled: ", body['caption_enabled'], type(body['caption_enabled']))
        caption_enabled = bool(body['caption_enabled'])
        
    print("Caption enabled: ", caption_enabled)
    
    # if 'queryStringParameters' in event:
    #     if 'topic' in event['queryStringParameters']:
    #         topic = event['queryStringParameters']['topic']
    #     if 'directory' in event['queryStringParameters']:
    #         directory_name = event['queryStringParameters']['directory']
    #     if 'script' in event['queryStringParameters']:
    #         script = event['queryStringParameters']['script']
    # else:
    #   if 'topic' in event:
    #     topic = event['topic']
    #   if 'directory' in event:
    #     directory_name = event['directory']
    #   if 'script' in event:
    #     script = event['script']
        
    if topic is None or directory_name is None or script is None:
        return {
            'statusCode': 400,
            'body': json.dumps({"error": "Missing topic or director or script"})
        }

    split_script_result = split_script(script)
        
    # process audio lambda
    invoke_lambda("process_audio", {"folder_name": directory_name, "script_array": split_script_result, "caption_enabled": caption_enabled})
        
    # process video lambda
    invoke_lambda("process_video", {"folder_name": directory_name, 
                                    "script": script, 
                                    "topic": topic,
                                    "caption_enabled": caption_enabled,
                                    "title": f"Create a title card for the plot: {topic}"})
    
    return {
        'statusCode': 200,
        'body': json.dumps({"folder_name": directory_name, "script_arr": split_script_result})
    }
