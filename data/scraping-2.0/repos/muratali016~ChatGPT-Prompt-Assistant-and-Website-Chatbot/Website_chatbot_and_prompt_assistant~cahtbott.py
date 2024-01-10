import openai
from openai import Completion
import speech_recognition as sr
import pyttsx3
import webbrowser
import time

# Set up the OpenAI API client
openai.api_key = "" 



def send_message(message):
    # Use the OpenAI API to get a response from ChatGPT
    response = Completion.create(
        engine="text-davinci-003",
        prompt=message,
        max_tokens=1024,
        temperature=0.5,
    )
    return response.get('choices')[0].get('text')







