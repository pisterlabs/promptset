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
from playsound import playsound


import config
import RPi.GPIO as GPIO
import sys

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

import openai
from elevenlabs import generate, play, stream, voices, save
from elevenlabs import set_api_key

msgs = ["sht", "ask", "spk", "done"]

client_process = None


# OpenAI API Key

api_key = config.api_key
elevenLabsAPiKey = config.elevenLabsAPiKey
voice_id = config.voice_id

isProcessing = False


start_time = 0


set_api_key(elevenLabsAPiKey)

thePrompt = "You're William Shakespeare, You tell people what you can describe on the image provided. Take into account common sense and always stay respectful. You're reviewing images from your own point of view, you are not aware of anything that happened after the year 1616 and you're staying true to what is historically known about Shakespeare's life. \n\nYou'll receive images one at a time, \n\nYou'll never answer with a question, this is a one time conversation with William.\n\nWhen you answer the user, you'll randomly choose 1  of the following 4 response patterns, keeping the same context.\n\n1) You'll answer with a short rhyme.\n2) You'll answer in period correct early Modern English, Elizabethan English.\n3) You answer from the point of view of one of the characters you've written about.\n4) You'll answer from a perspective of what it's like living in England in the 17th century.\n\n\nIf someone asks you a personal questions reply in a witty sarcastic manner.  \n\n It's very important that you begin each answer with a variation of this: \n 'Ok, this is what I see on the image ' "



# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

def save_log(message):
    with open("log.txt", "a") as log_file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_file.write(f"{timestamp} - {message}\n")

def write_text_on_image(image_path, text, position=(10, 10), font_size=5, font_color="white"):

    try:
        # Open the image
        with Image.open(image_path) as img:
            # Create a drawing context
            draw = ImageDraw.Draw(img)

            # Load a font
            #font = ImageFont.truetype("arial.ttf", font_size)

            # Add text to image
            draw.text(position, text, fill=font_color)

            # Save the image
            img.save(image_path)

    except IOError as e:
        print(f"Error opening or processing the image: {e}")

def getImageInfo(image_path):
    base64_image = encode_image(image_path)
    print("asking open ai for --->", {image_path})
    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
    }
    payload = {
      "model": "gpt-4-vision-preview",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": thePrompt
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "max_tokens": 1024
    }
    openAI_response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print(openAI_response.json())
    msg = openAI_response.json()
    return msg['choices'][0]['message']['content']

def create_video_from_image_and_audio(image_path, audio_path, output_video_path):
    try:
        command = [
            'ffmpeg',
            '-loop', '1',
            '-framerate', '1',
            '-i', image_path,
            '-i', audio_path,
            '-c:v', 'libx264',
            '-tune', 'stillimage',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            '-pix_fmt', 'yuv420p',
            output_video_path
        ]

        subprocess.run(command, check=True)
        print(f"Video created successfully: {output_video_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred: {e}")

def capture_image(uuidID, save_dir="/home/pi/openAI-rpi-11labs-test/captures"):
    # Ensure the save directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create a file name based on the current time
    file_name = uuidID + ".jpg"
    file_path = os.path.join(save_dir, file_name)

    # Capture the image
    with picamera.PiCamera() as camera:
        camera.resolution = (1280, 720)  # You can adjust the resolution
        camera.hflip = True
        camera.rotation = 90
        camera.start_preview()
        # Camera warm-up time
        print("warming camera")
        time.sleep(1)
        camera.capture(file_path)
        print(f"Image captured and saved as {file_path}")

    return file_path

def process_image(filename, uuidID):
  info = getImageInfo(filename)
  

  

  logInfo = filename + " ---> " + info + "\n\n"
  write_text_on_image(filename, logInfo)
  save_log(logInfo)
  print("generating audio with elevenLabs")
  audiogen = generate(text =  info, voice=voice_id)

  

  nameOf = uuidID
  
  input_audio_path = "audios/" + nameOf + '_answer.wav'
  print("playing msg \n\n")
  print("saving msg \n\n")
  save(audiogen, input_audio_path )
  
  
  return info , input_audio_path, audiogen


def triggered_function():

  playsound('/home/pi/openAI-rpi-11labs-test/shutter.wav')
  

  start_time = time.time()
  isProcessing = True
  print("shooting....")
  uuidID = str( uuid.uuid4() )
  
  captured_image_path = capture_image(uuidID)
  process = process_image(captured_image_path, uuidID)
  
  #create_video_from_image_and_audio(captured_image_path, process[1], 'videos/' + uuidID + ".mp4" )
  
  end_time = time.time()
  elapsed_time = end_time - start_time
  print("task completed for UUID--> " + uuidID + " in exactly " + str(elapsed_time) + " secs")
  
  play(process[2])
  
  isProcessing = False



if __name__ == "__main__":
    print("initializing shakespeare camera") 
    time.sleep(2)
    #triggered_function()
    time.sleep(1)
    
    GPIO.setmode(GPIO.BCM)  # Use Broadcom pin numbering
    GPIO.setup(14, GPIO.IN, pull_up_down=GPIO.PUD_UP)  # Button to GPIO17

try:
    while True:
        button_state = GPIO.input(14)
        if button_state == False and isProcessing == False:  # Button is pressed
            triggered_function()
            time.sleep(0.2)  # Add a small delay to debounce

finally:
    GPIO.cleanup()  # Clean up GPIO on normal exit