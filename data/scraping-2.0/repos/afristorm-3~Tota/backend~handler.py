#!/usr/bin/python3
"""
Contains handler functions for audio transcription,chatbot and text-to-audio conversion.

"""
import json
import base64
import openai
import io
import requests
from requests.exceptions import Timeout

openai.api_key = "ADD OPENAI API KEY HERE"

def transcribe_audio(audio_data):
    """
    Transcribes the given audio data using OpenAI's Whisper model.
    
    Args:
        audio_data (bytes): Raw audio data in bytes.
    
    Returns:
        str: The transcribed text.
    """
    with io.BytesIO(audio_data) as audio_file:
        audio_file.name = "audio.mp3"
        response = openai.Audio.transcribe(model="whisper-1", file=audio_file, language="en")
    
    transcription = response["text"]
    return transcription


def generate_chat_completion(messages):
    """
    Generates a chat completion response based on the given messages.
    
    Args:
        messages (list): List of message objects containing role and content.
        
    Returns:
        str: The generated chat response.
    """
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=100, 
    )
    return response.choices[0].message["content"]


def generate_audio(generated_text):
    """
    Generates audio from the generated text using ElevenLabs Text-to-Speech API.
    
    Args:
        generated_text (str): The generated text.
        
    Returns:
        str: Base64 encoded audio content.
    """
    api_key = "ADD ELEVENLAB API KEY HERE"
    voice_id = "ADD VOICE ID HERE"
    
    data = {
        "text": generated_text,
        "voice_settings": {
            "stability": 0,
            "similarity_boost": 0
        }
    }
   
    url = f'https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?api_key={api_key}'
    headers = {
        'accept': 'audio/mpeg',
        'Content-Type': 'application/json'
    }

    try:
        response = requests.post(url, headers=headers, json=data, timeout=15)
    except Timeout:
        return None
    return base64.b64encode(response.content).decode('utf-8')


def handler(event, context):
    try:
        body = json.loads(event["body"])

        if 'audio' in body:
            audio_base64 = body["audio"]
            audio_data = base64.b64decode(audio_base64.split(",")[-1])
            transcription = transcribe_audio(audio_data)
            message_objects = body['messages'] + [{"role": "user", "content": transcription}]
        elif 'text' in body:
            transcription = body['text']
            message_objects = body['messages']
        else:
            raise ValueError("Invalid request format. Either 'audio' or 'text' key must be provided.")

        generated_text = generate_chat_completion(message_objects)
        
        is_audio_response = body.get('isAudioResponse', False)

        if is_audio_response:
            generated_audio = generate_audio(generated_text)
        else:
            generated_audio = None

        response = {
            "statusCode": 200,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps(
                {"transcription": transcription, "generated_text": generated_text, "generated_audio": generated_audio}),
        }
        return response

    except ValueError as ve:
        import traceback
        print(traceback.format_exc())
        print(f"ValueError: {str(ve)}")
        response = {
            "statusCode": 400,
            "body": json.dumps({"message": str(ve)}),
        }
        return response
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        print(f"Error: {str(e)}")
        response = {
            "statusCode": 500,
            "body": json.dumps({"message": "An error occurred while processing the request."}),
        }
        return response

