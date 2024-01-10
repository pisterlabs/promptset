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

openai.api_key = os.environ["OPENAI_API_KEY"]
def generate_image_with_dalle(prompt,filename):
    try:
        response = openai.Image.create(
            # model="dall-e-2",
            # quality="standard",
            model="dall-e-3",
            quality="hd",
            prompt=prompt,
            n=1
            # specify other parameters as needed, such as size
        )
        image_url = response['data'][0]['url']
        # Assuming the response contains a direct link to the image or the image data
        img_response = requests.get(image_url)
        img_response.raise_for_status()

        # Upload the image to S3
        bucket.put_object(Key=filename, Body=img_response.content)
        print(f"Image saved to S3 bucket '{bucket_name}' with filename '{filename}'")
        return True
    except Exception as e:
        print(f"An error occurred: {e}")
        return False

def parse_script(script):
    # Implement logic to split the script into scenes
    # This is a placeholder function
    scenes = script.split("Scene")  # Example, depends on your script format
    return scenes

def process_video(s3_location, script, topic):
    scenes = parse_script(script)
    i = 1
    for scene in scenes:
        prompt = f"Overarching story: {topic}; Scene description: {scene}"  # Create a suitable prompt for DALL·E
        filename = f"{s3_location}/video/{i}.png"
        generate_image_with_dalle(prompt, filename)
        i+=1

def lambda_handler(event, context):
    folder_name = event['folder_name']
    script = event['script']
    title = event['title']
    topic = event['topic']
    caption_enabled = event['caption_enabled']
    print("Caption enabled: ", caption_enabled)

    # create title card:
    generate_image_with_dalle(title, f"{folder_name}/video/0.png")

    # video
    process_video(folder_name, script, topic)
    
    # invoke combine video
    invoke_lambda('combine_video', {"folder_name": folder_name, "caption_enabled": caption_enabled})
    
    return {
        'statusCode': 200,
        'body': json.dumps({"folder_name": folder_name})
    }
