import boto3
import openai
import pyttsx3
import cv2
import tempfile
import os
import numpy as np
import imutils
import time
from gtts import gTTS
import pygame
from pygame import mixer

aws_access_key = 'AKIAQJYDURIUH7ASYGEB'
aws_secret_key = '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'
openai.api_key = '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'

def detect_labels(image_bytes):
    client = boto3.client('rekognition', aws_access_key_id=aws_access_key, aws_secret_access_key=aws_secret_key, region_name='ap-south-1')
    response = client.detect_labels(Image={'Bytes': image_bytes})
    label_names = [label['Name'] for label in response['Labels']]
    return label_names

# Function to generate response using OpenAI GPT model
def generate_response(prompt):
    response = openai.ChatCompletion.create(
              model="gpt-3.5-turbo-0301",
              messages=[{"role": "system", "content": 'You are a helpful assistant who is accompanying a blind person'},
                        {"role": "user", "content": prompt}
              ])
    return response.choices[0].message['content'].strip()

# Initialize webcam capture
camera = cv2.VideoCapture(0)  # 0 indicates the default webcam

# Process and display image
def process_and_display_image():
    top_labels = []
    for _ in range(5):
        ret, frame = camera.read()
        if not ret:
            break
        
        resized_frame = imutils.resize(frame, width=1000, height=1800)
        
        # Save the image to a temporary file
        temp_image = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        cv2.imwrite(temp_image.name, resized_frame)
        image_path = temp_image.name
        temp_image.close()  # Close the file handle
        
        labels = detect_labels(open(image_path, 'rb').read())
        top_labels.extend(labels)
        
        os.remove(image_path)  # Now you can safely remove the file
        time.sleep(1)  # Wait for 1 second before fetching the next image
    
    top_labels = list(set(top_labels))  # Remove duplicate labels
    prompt = "Image labels: " + ", ".join(top_labels) + "\n"
    print(prompt)
    user_prompt = "i have given you some keywords which are extracted from aws rekognition service and from them you have to describe a scenario out of them keep the tone and manner in a way like you are describibg a scenario to a blind person. with compassionate keep the description short and easy and also talk like you with that person " + "\n"  
    prompt += user_prompt
    print(prompt)
    
    response_text = generate_response(prompt)
    print(response_text)
    return response_text

# Run the processing loop
response_text = process_and_display_image()

# Release the camera
camera.release()

# Convert response text to speech audio
tts = gTTS(text=response_text, lang='en')
audio_file = 'response_audio.mp3'
tts.save(audio_file)

# Initialize the mixer module of pygame
pygame.mixer.init()

# Play the response audio
pygame.mixer.music.load(audio_file)
pygame.mixer.music.play()

# Wait for the audio to finish playing
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)

# Cleanup pygame resources
pygame.mixer.quit()
