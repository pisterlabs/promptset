import sys
import csv
import os
import base64
import time
import datetime
import requests
from asterisk.agi import AGI  # Asterisk Gateway Interface for IVR interactions
from google.cloud import speech  # Google Cloud Speech-to-Text
from google.cloud.speech import enums
from google.cloud.speech import types
from openai import OpenAI  # OpenAI API

def record_and_transcribe(agi, language, sample_rate, credentials_path, wav_file):
    """
    Plays a WAV file to the caller, records their response, and transcribes it.

    :param agi: Asterisk AGI interface.
    :param language: Language code for transcription.
    :param sample_rate: Sample rate of the audio.
    :param credentials_path: Path to Google Cloud credentials.
    :param wav_file: Path to the WAV file to be played.
    :return: Transcribed text of the caller's response.
    """
    # Play the WAV file if specified
    if wav_file:
        agi.stream_file(wav_file)

    # Start recording the caller's response
    agi.record_file("caller_response", "wav", escape_digits="", timeout=10000)  # 10 seconds timeout
    file_path = "/path/to/caller_response.wav"  # Update with actual path

    # Load the audio into memory
    with open(file_path, 'rb') as audio_file:
        audio_content = audio_file.read()

    # Transcribe the audio file using Google Cloud Speech-to-Text
    client = speech.SpeechClient.from_service_account_json(credentials_path)
    audio = types.RecognitionAudio(content=audio_content)
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=sample_rate,
        language_code=language,
    )

    # Detects speech in the audio file and returns the transcription
    response = client.recognize(config=config, audio=audio)
    return response.results[0].alternatives[0].transcript if response.results else ""

def check_ivr_response(ivr_response, allowed_responses):
    """
    Uses OpenAI to analyze the IVR response and determine the intent.

    :param ivr_response: The response text from the caller.
    :param allowed_responses: List of allowed responses or intents.
    :return: Intent or clarifying question identified by OpenAI.
    """
    # Load API key from an environment variable
    api_key = os.getenv('OPENAI_API_KEY')
    client = OpenAI(api_key=api_key)

    # Create the thread for OpenAI to analyze the response
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "system",
                "content": "You are receiving a transcription from a customer service IVR your job is to determine the intent of the customer and pass the intent back to the python script that called this chat from a list of allowed intents. You should only reply with the intent or if needed ask a  up to two clarifying questions. Clarifying questions should start with "!" so the python script can know its not the intent but you are passing back but a clarifying question. or in the event you cannot determine the intent reply defaultdropout. The allowed intents are will be passed to you with the transcription."
                
            },
            {
                "role": "user",
                "content": f"IVR response: '{ivr_response}'. Allowed intents are: {', '.join(allowed_responses)}."
            }
        ]
    )

    # Retrieve the assistant ID for 'IVR assistant level one'
    assistant_id = "your_assistant_id"  # Replace with actual assistant ID

    # Run the thread with the assistant and retrieve the response
    try:
        run = client.beta.threads.runs.create(
            thread_id=thread.id,
            assistant_id=assistant_id
        )

        # Get the last message, which should be the response from OpenAI
        response = run.messages[-1]['content']
        return response.strip()

    except Exception as e:
        print(f"Error: {e}")
        return None

def generate_speech_with_elevenlabs(text, xi_api_key, voice_id="default_voice_id"):
    """
    Converts text to speech using Eleven Labs' API.

    :param text: Text to be converted to speech.
    :param xi_api_key: API key for Eleven Labs.
    :param voice_id: ID of the voice to be used.
    :return: File path of the generated speech audio.
    """
    url = "https://api.elevenlabs.io/v1/text-to-speech/{}".format(voice_id)
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": xi_api_key
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {
            "stability": 0.5,
            "similarity_boost": 0.5
        }
    }
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 200:
        with open('response_audio.mp3', 'wb') as f:
            f.write(response.content)
        return 'response_audio.mp3'
    return None

def main():
    """
    Main function to handle the IVR call flow.
    """
    agi = AGI()

    # Configuration Variables
    GOOGLE_CLOUD_SPEECH_CREDENTIALS = '/path/to/your/credentials.json'
    XI_API_KEY = "your_elevenlabs_api_key"
    LANGUAGE = 'en-US'
    SAMPLE_RATE = 8000
    WAV_FILE = "welcome_message.wav"  # Update with actual WAV file path

    try:
        # Record the caller's response and transcribe
        transcribed_text = record_and_transcribe(agi, LANGUAGE, SAMPLE_RATE, GOOGLE_CLOUD_SPEECH_CREDENTIALS, WAV_FILE)
        agi.verbose("Transcribed Text: " + transcribed_text)

        # Analyze intent with OpenAI
        intent_or_question = check_ivr_response(transcribed_text, ["pass_to_agent", "book_an_engineer", "make_a_complaint", "cancel_appointment", "default_dropout"])

        # Check if the response is a clarifying question
        if intent_or_question.startswith("!"):
            # Handle clarifying question
            clarifying_question = intent_or_question[1:]  # Remove '!' from the beginning
            speech_file = generate_speech_with_elevenlabs(clarifying_question, XI_API_KEY)  # Generate speech for the clarifying question
            if speech_file:
                agi.stream_file(speech_file.replace('.mp3', ''))  # Play the clarifying question

            # Record and analyze response to the clarifying question
            follow_up_text = record_and_transcribe(agi, LANGUAGE, SAMPLE_RATE, GOOGLE_CLOUD_SPEECH_CREDENTIALS, None)
            intent = check_ivr_response(follow_up_text, ["pass_to_agent", "book_an_engineer", "make_a_complaint", "cancel_appointment", "default_dropout"])
        else:
            intent = intent_or_question  # Directly use the intent if no clarifying question

        # Route call based on intent to the respective hunt group
        if intent == "pass_to_agent":
            agi.set_extension("201")  # Route to hunt group 201
        elif intent == "book_an_engineer":
            agi.set_extension("202")  # Route to hunt group 202
        elif intent == "make_a_complaint":
            agi.set_extension("203")  # Route to hunt group 203
        elif intent == "cancel_appointment":
            agi.set_extension("204")  # Route to hunt group 204
        elif intent == "default_dropout":
            agi.set_extension("205")  # Route to hunt group 205 (default/fallback)
        else:
            agi.verbose("Unrecognized intent received.")
            agi.set_extension("205")  # Fallback to a default extension if intent is not recognized

    except Exception as e:
        agi.verbose("Error: " + str(e))

    finally:
        agi.hangup()

if __name__ == "__main__":
    main()
