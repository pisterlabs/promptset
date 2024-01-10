import json
import requests
import random
from openai import OpenAI
import traceback
import os
import re

bot_id = os.environ['BOT_ID']
groupme_post_message_url = "https://api.groupme.com/v3/bots/post"
openai_api_key = os.environ['OPENAI_API_KEY']
omar_user_id = "60388229"

def generate_response_and_send_message(style_message, input_message):

    client = OpenAI(
        api_key = openai_api_key
    )

    try:
        message = "Hey Cracka!"

        # Generate a response using OpenAI
        chat_completion = client.chat.completions.create(
            messages= [
                {
                    "role": "system",
                    "content": style_message,
                },
                {
                    "role": "user",
                    "content": input_message,
                }
            ],
            model="gpt-3.5-turbo-1106",
            max_tokens=150,
            temperature=0.9,
        )

        # Extract the text from the response
        message = chat_completion.choices[0].message.content

        # Prepare the response message
        response_message = {
            "bot_id": bot_id,
            "text": message
        }

        # Send the message
        requests.post(groupme_post_message_url, json=response_message)

    except Exception as e:
        print(f"Error generating response: {str(e)}")
        traceback.print_exc()
        message = "Sorry, I couldn't process that."

def send_message(message):
    response_message = {
        "bot_id": bot_id,
        "text": message
    }

    # Send the message
    requests.post(groupme_post_message_url, json=response_message)

def lambda_handler(event, context):
    # Your GroupMe bot ID

    print("event:", event)

    # Check if message contains "Luke Butt" using regex
    if event['sender_id'] != bot_id and event['sender_type'] != "bot":
        random_numer = random.randint(0, 100)

        if re.search(r"uncle sherwin", event['text'].lower()):
            generate_response_and_send_message("You are helpful assistant that will answer every question. Your name is Uncle Sherwin.", event['text'])

        # Check if the sender is not the bot itself
        # Assuming the message data includes sender_type or sender_id
        elif random_numer > 99:
            # GroupMe API endpoint for posting messages

            # Send the message
            generate_response_and_send_message("Respond to the user's input with humor. Use light-hearted and witty humor, appropriate for a general audience. " + \
                                               "Incorporate playful language, puns, and clever wordplay where possible. Aim to entertain and amuse without being " + \
                                               "offensive. If the user's input is about a specific topic, make humorous connections or jokes related to that topic. " + \
                                               "Avoid sarcasm or irony that could be misunderstood. Keep the tone friendly and engaging.", \
                                               event['name'] + "said: " + event['text'])

    return {
        'statusCode': 200,
        'body': json.dumps('OK')
    }