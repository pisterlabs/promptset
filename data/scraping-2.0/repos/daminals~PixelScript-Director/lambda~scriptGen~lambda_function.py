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


# def create_bucket(bucket_name):
#     # create bucket inside gpt3-video-scripts bucket
#     new_bucket = s3_client.create_bucket(
#         Bucket=bucket_name,
#     )
#     bucket.put_object(Key=bucket_name, Body=new_bucket)
#     return new_bucket

openai.api_key = os.environ["OPENAI_API_KEY"]
def generate_video_script(director, topic):
    response = openai.ChatCompletion.create(
        model="ft:gpt-3.5-turbo-1106:personal::8R6vhAys",
        messages=[
            {"role": "system", "content": f"MovieAI is a masterful filmmaker which emulates famous directors in order to create new twists on old concepts and films. It will return the response in the following format: Seperate scenes by using \"Scene 1\n\" where 1 corresponds to the scene number and clearly indicate who is currently speaking by utilizing \"Voice 1:\n\" where 1 corresponds to the number of the character currently speaking"},
            {"role": "user", "content": f"In the style of {director}, write {topic}. Seperate scenes by using \"Scene 1\n\" where 1 corresponds to the scene number and clearly indicate who is currently speaking by utilizing \"Voice 1:\n\" where 1 corresponds to the number of the character currently speaking"}
        ],
    )
    return response.choices[0]['message']['content']

def generate_image_with_dalle(prompt,filename):
    try:
        response = openai.Image.create(
            prompt=prompt,
            # specify other parameters as needed, such as size
        )
        image_url = response['data'][0]['url']
        # Assuming the response contains a direct link to the image or the image data
        img_response = requests.get(image_url)
        img_response.raise_for_status()

        # Upload the image to S3
        bucket.put_object(Key=filename, Body=img_response.content)
        # print(f"Image saved to S3 bucket '{bucket_name}' with filename '{filename}'")
        return True
    except Exception as e:
        # print(f"An error occurred: {e}")
        return False

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


def process_audio(directory_name, script_arr):
    voiceIDs = [
        "Matthew",
        "Russell",
        "Emma",
        "Ivy",
        "Kendra",
        "Kimberly",
        "Joey",
        "Justin"
    ]
    neural_voices = [
      "Nicole",
      "Amy",
      "Arthur",
      "Joanna",
      "Kevin",
      "Stephen",
      "Salli"
    ]
    
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
        except:
            print(f"{voices[voice]} is a failure")
        line_num += 1


def tts(text, voiceId, filename):
    response = polly.synthesize_speech(
        Text=text,
        OutputFormat="mp3",
        VoiceId=voiceId
    )
    stream = response["AudioStream"]
    bucket.put_object(Key=filename, Body=stream.read())


def parse_script(script):
    # Implement logic to split the script into scenes
    # This is a placeholder function
    scenes = script.split("Scene")  # Example, depends on your script format
    return scenes

def process_video(s3_location, script):
    scenes = parse_script(script)
    for i, scene in enumerate(scenes):
        prompt = f"Scene description: {scene}"  # Create a suitable prompt for DALL·E
        filename = f"{s3_location}/video/{i}.png"
        generate_image_with_dalle(prompt, filename)

def generate_foldername(director):
    # generate a random 8 digit number
    num = random.randint(10000000, 99999999)
    director = director.replace(" ", "-")
    return f"{director}-{num}"

def lambda_handler(event, context):
    topic = None 
    director = None
    
    if 'queryStringParameters' in event:
        if 'topic' in event['queryStringParameters']:
            topic = event['queryStringParameters']['topic']
        if 'director' in event['queryStringParameters']:
            director = event['queryStringParameters']['director']
    else:
      if 'topic' in event:
        topic = event['topic']
      if 'director' in event:
        director = event['director']
        
    if topic is None or director is None:
        return {
            'statusCode': 400,
            'body': json.dumps({"error": "Missing topic or director"})
        }
    video_script = generate_video_script(director, topic)
    split_script_result = split_script(video_script)
    
    directory_name = generate_foldername(director)
    
    # # process audio lambda
    # invoke_lambda("process_audio", {"folder_name": directory_name, "script_array": split_script_result})
        
    # # video
    # invoke_lambda("process_video", {"folder_name": directory_name, 
    #                                 "script": video_script, 
    #                                 "topic": topic,
    #                                 "title": f"Create a title card for the plot: {topic}"})
    
    # res_body = {
    #     "folder_name": directory_name,
    #     "script": video_script
    # }
    
    
    return {
        'statusCode': 200,
        # 'body': json.dumps({"folder_name": directory_name})
        'body': json.dumps({"folder_name": directory_name, "script": video_script})
    }
