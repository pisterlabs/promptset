import os
import ssl
import cv2
import base64
import openai

from dotenv import load_dotenv
from openai import OpenAI
from pytube import YouTube
from IPython.display import Image

load_dotenv()

client = OpenAI()

openai.api_key = os.environ["OPENAI_API_KEY"]

ssl._create_default_https_context = ssl._create_unverified_context

print(openai.api_key)

video_url = 'https://www.youtube.com/watch?v=kQ_7GtE529M'
path_to_save_video = 'images/video'


def download_video(video_url, path_to_save_video):
    yt = YouTube(video_url)
    video_stream = yt.streams.get_highest_resolution()
    video_stream.download(output_path=path_to_save_video)
    print(f"Downloaded: {yt.title}")


# runs only once to download the video
# download_video(video_url, path_to_save_video)

video = cv2.VideoCapture("images/video/Douchebag_Bison.mp4")

base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success or frame is None:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

print("Total Frames: ", len(base64Frames))

video.release()

Image(data=base64.b64decode(base64Frames[100].encode("utf-8")))

#  Create a short voiceover script for a  commentator. Only include the narration.

prompt = """
These are frames of a video. 
Create a short voiceover script for a nature documentary narrator. 
Only include the narration.
"""
images = map(
    lambda x: {"image": x, "resize": 768},
    base64Frames[0:100:10]
)

result = client.chat.completions.create(
    model='gpt-4-vision-preview',
    max_tokens=4096,
    messages=[{
        'role': 'user',
        'content': [prompt, *images]
    }]
)

print("Narrative : ", result.choices[0].message.content, "\n")


def get_response(prompt, images):
    result = client.chat.completions.create(
        model='gpt-4-vision-preview',
        max_tokens=4096,
        messages=[{
            'role': 'user',
            'content': [prompt, *images]
        }]
    )
    return result.choices[0].message.content


# history is the previous script
def get_prompt(history):
    base_prompt = """
    These are frames of a video. 
    Continue the short voiceover script for a nature documentary narrator. 
    Only include the narration.

    Script for the previous frames: {history}

    Script for the new frames:"""

    return base_prompt.format(history=history)


MAX_NUM = 50


def get_images(i, image_list):
    init = i * MAX_NUM
    final = (i + 1) * MAX_NUM
    return map(
        lambda x: {"image": x, "resize": 768},
        image_list[init:final:20]
    )

n = len(base64Frames) // 50
history = ''

for i in range(n):
    images = get_images(i, base64Frames)
    prompt = get_prompt(history)
    response = get_response(prompt, images)
    history += '\n\n' + response

print("narrative ", history)

# I want to save the narrative as a text file
with open('narrative.txt', 'w') as f:
    f.write(history)


# converting the script to audio
response = client.audio.speech.create(
  model="tts-1",
  voice="onyx",
  input=history
)
response.stream_to_file("narrative.mp3")

from moviepy.editor import VideoFileClip, AudioFileClip

video_clip = VideoFileClip("original_video.mp4")
audio_clip = AudioFileClip("comments.mp3")

final_clip = video_clip.set_audio(audio_clip)
final_clip.write_videofile(
    'final_video.mp4',
    codec='libx264',
    audio_codec='aac',
    fps=30.0
)
