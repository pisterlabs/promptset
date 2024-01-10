import os
import azure.cognitiveservices.speech as speechsdk
import openai
import urllib.request
import json
import os
import ssl



# Set up Azure Speech-to-Text and Text-to-Speech credentials

speech_key = "753db2da843e4e2c81d36500b981f159"
service_region, endpoint = "eastus", "https://eastus.api.cognitive.microsoft.com/sts/v1.0/issuetoken"

speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=service_region)
# Set up Azure Text-to-Speech language 
speech_config.speech_synthesis_language = "en-NZ"
# Set up Azure Speech-to-Text language recognition
speech_config.speech_recognition_language = "en-NZ"

# Set up the voice configuration
speech_config.speech_synthesis_voice_name = "en-NZ-MollyNeural"
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)

openai.api_key = "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"
#openai.api_base = ""
#openai.api_type = 'azure'
#openai.api_version = '2023-07-01-preview'


# Define a function to generate a response using Azure OpenAI
def openai_generate_response(prompt):
    try:
        response = openai.ChatCompletion.create(        
            engine="gpt-3",
            messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]     
        )
    ##  print(response)
        return response.choices[0].message.content
    except openai.error.InvalidRequestError as e:
        print(f"Invalid request error: {e}")
    except openai.error.AuthenticationError as e:
        print(f"Authentication error: {e}")
    except openai.error.APIConnectionError as e:
        print(f"API connection error: {e}")
    except openai.error.OpenAIError as e:
        print(f"OpenAI error: {e}")



# Define the speech-to-text function
def speech_to_text():
    # Set up the audio configuration
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)

    # Create a speech recognizer and start the recognition
    speech_recognizer = speechsdk.SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
    print("Say something...")

    result = speech_recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        return result.text
    elif result.reason == speechsdk.ResultReason.NoMatch:
        return "Sorry, I didn't catch that."
    elif result.reason == speechsdk.ResultReason.Canceled:
        return "Recognition canceled."

# Define the text-to-speech function
def text_to_speech(text):
    try:
        result = speech_synthesizer.speak_text_async(text).get()
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            print("Text-to-speech conversion successful.")
            return True
        else:
            print(f"Error synthesizing audio: {result}")            
            return False
    except Exception as ex:
        print(f"Error synthesizing audio: {ex}")
        return False




# Main program loop
while True:
    # Get input from user using speech-to-text
    user_input = speech_to_text()
    print(f"You said: {user_input}")

    # Generate a response using OpenAI
    prompt = f"Q: {user_input}, answer in 1 sentence, use conversational english type of response\nA:"
    response = openai_generate_response(prompt)
    #response = user_input
    print(f"AI says: {response}")

    # Convert the response to speech using text-to-speech
    text_to_speech(response)