import openai
import base64
import pygame
from pathlib import Path
from openai import OpenAI
import cv2
import time

# Wait for five seconds to give the user time to get ready
time.sleep(5)
# Open the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Read the image from the webcam
ret, frame = cap.read()

# Save the image
cv2.imwrite('webcam_image.jpg', frame)

# Release the webcam
cap.release()
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode()
#encode_image("webcam_image.jpg")
base64_image = encode_image("webcam_image.jpg")

# Past the image to the openai vision API
response = openai.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image?",
                },
                {
                    "type": "image_url",
                    "image_url": f"data:image/jpeg;base64,{base64_image}",
                }
            ]
        }
    ],
    max_tokens=300, 
)
# Load the description of the image from the response into a variable
imagecontent = response.choices[0].message.content
print(imagecontent)


client = OpenAI()
# Call openai Text to speech model
speech_file_path = Path(__file__).parent / "speech.mp3"
response = client.audio.speech.create(
  model="tts-1",
  voice="shimmer",
  input=imagecontent
)
# Save the response to a mp3 file
response.stream_to_file(speech_file_path)

# Initialize pygame to play the MP3 file individuals man man
pygame.init()

# Load the MP3 file
pygame.mixer.init()
pygame.mixer.music.load(speech_file_path)  # Path to your MP3 file
# Play the MP3 file
pygame.mixer.music.play()

# Keep the script running until the music is playing
while pygame.mixer.music.get_busy():
    pygame.time.Clock().tick(10)
