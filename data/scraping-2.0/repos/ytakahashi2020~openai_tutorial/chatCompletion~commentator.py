import time
import cv2
import base64
import openai
from moviepy.editor import VideoFileClip, AudioFileClip

# Initialize OpenAI client
client = openai.OpenAI()

# Load the video
video = cv2.VideoCapture("./football.mov")

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
                *map(lambda x: {"image": x, "resize": 768},
                     base64Frames[0::25]),
            ]
        }
    ],
    max_tokens=500,
)

# Print the response content
print(response.choices[0].message.content)
time.sleep(2)
speech_file_path = "football.mp3"
response = client.audio.speech.create(
    model="tts-1",
    voice="onyx",
    input=response.choices[0].message.content
)

response.stream_to_file(speech_file_path)


video_clip = VideoFileClip("football.m4v")
audio_clip = AudioFileClip("football.mp3")
final_clip = video_clip.set_audio(audio_clip)
final_clip.write_videofile(
    "football_with_commentator.m4v", codec='libx264', audio_codec='aac')
video_clip.close()
audio_clip.close()
final_clip.close()
