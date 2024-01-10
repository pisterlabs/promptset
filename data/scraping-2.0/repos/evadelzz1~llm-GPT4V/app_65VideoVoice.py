from dotenv import load_dotenv
import openai
import base64
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)
    
# Initialize OpenAI client
client = openai.OpenAI()

# Load the video
# https://pixabay.com/ko/videos/축구-스포츠-무인-비행기-158633/
video = cv2.VideoCapture("./data/football.mp4")

# Calculate video length
length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
fps = video.get(cv2.CAP_PROP_FPS)
video_length_seconds = length / fps

print(f'Video length: {video_length_seconds:.2f} seconds')

# Read frames and encode to base64
base64Frames = []
while video.isOpened():
    success, frame = video.read()
    if not success:
        break
    _, buffer = cv2.imencode(".jpg", frame)
    base64Frames.append(base64.b64encode(buffer).decode("utf-8"))

# Release the video object
video.release()
print(len(base64Frames), "frames read.")

# Create OpenAI chat completion
response = client.chat.completions.create(
    model="gpt-4-vision-preview",
    messages=[
        {
            "role": "user", 
            "content": [
                f"These are frames of a video. Create a short voiceover script in the style of a football commentator For {video_length_seconds:.2f} seconds. Only include the narration. Don't talk about the view",
                *map(lambda x: {"image": x, "resize": 768}, base64Frames[0::25]),
            ]
        }
    ],
    max_tokens=1000,
)

# Print the response content
print(response.choices[0].message.content)
import time
time.sleep(2)
speech_file_path = "./data/football.mp3"
response = client.audio.speech.create(
  model="tts-1",
  voice="onyx",
  input=response.choices[0].message.content
)

response.stream_to_file(speech_file_path)


video_clip = VideoFileClip("./data/football.mp4")
audio_clip = AudioFileClip("./data/football.mp3")
final_clip = video_clip.set_audio(audio_clip)
final_clip.write_videofile("./data/football_with_commentator.m4v", codec='libx264', audio_codec='aac')
video_clip.close()
audio_clip.close()
final_clip.close()