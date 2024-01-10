import cv2
import base64
import time
import requests
import backoff
import os
from openai import OpenAI, RateLimitError
from IPython.display import Audio  # Only use this if you want to display the audio in a Jupyter notebook
from moviepy.editor import VideoFileClip, AudioFileClip

# Initialize the OpenAI client
client = OpenAI()

# Use OpenCV to extract frames
video = cv2.VideoCapture("data/kingdom.mov")

base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

video.release()
print(len(base64Frames), "frames read.")

# Craft prompt and send request to LLM for description
PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            "These are frames from a video that I want to upload. Generate a compelling description that I can upload along with the video.",
            *map(lambda x: {"image": x, "resize": 256}, base64Frames[0:600:50]),
        ],
    },
]
params = {
    "model": "gpt-4-vision-preview",
    "messages": PROMPT_MESSAGES,
    "max_tokens": 200,
}

result = client.chat.completions.create(**params)
description = result.choices[0].message.content

# Generate a voiceover script in the style of David Attenborough
PROMPT_MESSAGES = [
    {
        "role": "user",
        "content": [
            "These are frames of a video. Create a short voiceover script in the style of David Attenborough. Only include the narration.",
            *map(lambda x: {"image": x, "resize": 256}, base64Frames[0:600:60]),
        ],
    },
]
params = {
    "model": "gpt-4-vision-preview",
    "messages": PROMPT_MESSAGES,
    "max_tokens": 500,
}

@backoff.on_exception(backoff.expo, RateLimitError)
def completions_with_backoff(**kwargs):
    response = client.chat.completions.create(**kwargs)
    return response

#result = client.chat.completions.create(**params)
result = completions_with_backoff(**params)
voiceover_script = result.choices[0].message.content

# Pass the script to the TTS API to generate an mp3 of the voiceover
response = requests.post(
    "https://api.openai.com/v1/audio/speech",
    headers={
        "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
    },
    json={
        "model": "tts-1-1106",
        "input": voiceover_script,
        "voice": "onyx",
    },
)

audio = b""
for chunk in response.iter_content(chunk_size=1024 * 1024):
    audio += chunk

# Save the audio as an mp3 file
with open("output_audio.mp3", "wb") as audio_file:
    audio_file.write(audio)

# Combine the audio with the video using moviepy
video_clip = VideoFileClip("data/kingdom.mov")
audio_clip = AudioFileClip("output_audio.mp3")
video_clip = video_clip.set_audio(audio_clip)

# Save the final video with the voiceover
video_clip.write_videofile("output_video_with_voiceover.mp4", codec="libx264", audio_codec="aac")

# Display the audio in a Jupyter notebook (remove if not using Jupyter)
Audio(audio)

