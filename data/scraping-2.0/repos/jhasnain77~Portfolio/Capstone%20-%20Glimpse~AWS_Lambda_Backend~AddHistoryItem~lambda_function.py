import json
import math
import uuid
import os
import sys

# Get the absolute path to the directory containing the Lambda function code
function_path = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the venv directory
venv_path = os.path.join(function_path, 'env/Lib/site-packages')

# Add the venv directory to the system path
sys.path.insert(0, venv_path)

import openai
import boto3

dynamodb = boto3.resource('dynamodb')
openai.api_key = "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"

def add_history_item(event, context):

    print(event)

    user_uuid = event['uuid']
    image = event['image']
    song_title = event['song_title']
    artist_name = event['artist_name']
    spotify_URL = event['spotify_URL']
    apple_URL = event['apple_URL']
    rating = event.get('rating', None)
    message_uuid = event.get('message_uuid', None)

    # Check if the user's history table exists
    table_name = user_uuid + '-history'
    try:
        table = dynamodb.Table(table_name)
    except:
        return {"statusCode": 404, "body": "User history table not found."}

    # Generate a new history item number
    response = table.scan(Select='COUNT')
    number = response['Count'] + 1
    
    # Add the new history item to the table
    history_item = {'number': number, 'song_title': song_title, 'artist_name': artist_name, 'spotify_URL': spotify_URL, 'apple_URL': apple_URL, 'image': image}
    
    if rating is not None:
        history_item['rating'] = rating

        # If a message UUID is present, provide feedback to the OpenAI model
        if message_uuid is not None:
            if rating == 1:
                feedback = "This was a great response!"
            elif rating == 0:
                feedback = "The song recommendation didn't fit the tags."
            else:
                feedback = None

            if feedback is not None:
                openai.Completion.create(
                    model="text-davinci-003",
                    prompt=f"feedback on message {message_uuid}: {feedback}",
                    temperature=0.7,
                    max_tokens=100,
                    n=1,
                    stop=None,
                )
    
    table.put_item(Item=history_item)

    # Return the new history item
    return {"statusCode": 200, "body": {"number": number, "song_title": song_title, "artist_name": artist_name, "spotify_URL": spotify_URL, "apple_URL": apple_URL, "rating": rating, "image": image}}
