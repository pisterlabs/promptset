from IPython.display import display, Image, Audio

import cv2  # We're using OpenCV to read video, to install !pip install opencv-python
import base64
import time
from openai import OpenAI
import os
import requests

client = OpenAI()
video = cv2.VideoCapture("data/encara-messi.mp4")

base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

video.release()
print(len(base64Frames), "frames read.")
PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
                "These are frames from a soccer video that I need help determining who scored as I am impaired. Which player scored? Make a best guess of the jersey number of the player who scored if you cannot determine for certain. Provide events in a numerically ordered list fashion. Be short and to the point and be sure to include the teams playing and the current scores after major events.",
            *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::75]),
        ],
    },
]
params = {
    "model": "gpt-4-vision-preview",
    "messages": PROMPT_MESSAGES,
    "max_tokens": 200,
}

result = client.chat.completions.create(**params)
print(result.choices[0].message.content)
