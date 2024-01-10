from flask import current_app as app
import openai
import os
import base64

def configure_openai_api():
    openai.api_key = app.config['OPENAI_API_KEY']

def call_openai_api(selected_frames):
    # Construct the payload with the frames as base64 encoded images
    print("Calling OpenAI API...")
    PROMPT_MESSAGES = [
        {
            "role": "user",
            "content": [
                "I have provided a number of images from a video of a golf swing. Review each image and determine what element of the golf swing it is. Using your knowledge of golf, provide useful feedback to the golfer as if you were the coach",
                *map(lambda x: {"image": x, "resize": 768}, selected_frames),
            ],
        },
    ]
    params = {
        "model": "gpt-4-vision-preview",
        "messages": PROMPT_MESSAGES,
        "max_tokens": 1000,
    }

    try:
        result = openai.ChatCompletion.create(**params)
        print (result)
        return result
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
