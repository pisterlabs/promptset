import base64
import requests
import os
import picamera
import time
import os
import datetime
import uuid
import subprocess
#create your config accordingly:
#api_key = "your_api_key_here"
#elevenLabsAPiKey = "your_elevenLabs_api_key_here"
#voice_id = "your_voice_id_here"

import config
import RPi.GPIO as GPIO
import sys

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

import openai
from elevenlabs import generate, play, voices, save
from elevenlabs import set_api_key

# OpenAI API Key

api_key = config.api_key
elevenLabsAPiKey = config.elevenLabsAPiKey
voice_id = config.voice_id

isProcessing = False


start_time = 0


set_api_key(elevenLabsAPiKey)

thePrompt = "Imagine thou art the esteemed William Shakespeare, revered bard, now in an age where hip hop reigns. Tasked with whimsically transcribing a verse from a renowned hip hop song in thine own Elizabethan style, first name the song, followed by '@' and the artist's name. Then, transcribe a verse, weaving it with eloquence akin to thy plays. Choose from four response styles: 1) A short rhyme, rich in wit and brevity; 2) In early Modern English, true to the Elizabethan era; 3) As a character from one of thy plays, embodying their persona; 4) From the perspective of a 17th-century Englishman, marveling at hip hop's peculiarities. Ensure responses are succinct and pointed, blending historical context with modern rhythm."

def getOpenAIText():
    print("asking...")

    response = openai.ChatCompletion.create(
      model="gpt-4",
      messages=[
        {
          "role": "system",
            "content": thePrompt 
    }
      ],
      temperature=1,
      max_tokens=256,
      top_p=1,
      frequency_penalty=0,
      presence_penalty=0
    )

    firstResult = response.choices[0].message.content
    return firstResult

if __name__ == "__main__":
    rhyme = getOpenAIText()
    audiogen = generate(text = thePrompt, voice=voice_id)
    print("playing")
    play(audiogen)
    print("played")