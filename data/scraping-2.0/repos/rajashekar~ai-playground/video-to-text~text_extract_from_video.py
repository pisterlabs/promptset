# This will extract text from video and save it in a text file
# Usage: python text_extract_from_video.py <video_path>
# Example: python text_extract_from_video.py video.mp4
# Note: This will save the text file in the same directory as the video file
# Uses open ai whisper API to extract text from audio
# https://beta.openai.com/docs/api-reference/transcriptions/create
# set OPENAI_API_KEY environment variable to your API key
# https://beta.openai.com/docs/developer-quickstart/your-api-keys

import sys
import os

from moviepy.editor import VideoFileClip
from pydub import AudioSegment
import openai

# Get video path from arguments
video_clip_path = sys.argv[1]
v_path = ' '.join(video_clip_path.split(".")[:-1])

# Step 1: Extract Audio from Video
video_clip = None
print("Extracting audio from video")
try:
    video_clip = VideoFileClip(video_clip_path)
    print(f"Video has {video_clip.duration // 60} minutes")
except Exception as e:
    print("its not a video file")
if video_clip is None and video_clip_path.split(".")[-1] == "mp4":
    print("trying to extract audio from mp4 file")
    audio = AudioSegment.from_file(video_clip_path, "mp4")
    audio.export(f"{v_path}.mp3", format="mp3")
else:
    print(f"it is not a mp4 file, it is a {video_clip_path.split('.')[-1]} file")

# Step 2: Save Audio to a mp3 file
if video_clip and not os.path.exists(f"{v_path}.mp3"):
    print("Saving audio to mp3 file")
    video_clip.audio.write_audiofile(f"{v_path}.mp3")
else:
    print("Audio file already exists")

# Step 3: Check duration of audio file
audio = AudioSegment.from_mp3(f"{v_path}.mp3")

# Duration of the mp3 file
print(f"Audio has {len(audio) // (60 * 1000)} minutes")

segments = 0
# if audio duration is greater than 10 minutes then split it into 10 minutes chunks
if len(audio) // (60 * 1000) > 10:
    print("Splitting audio into 10 minutes chunks")
    split_audio = True
    # Length of audio to split, in milliseconds
    length_ms = 10 * 60 * 1000  # 10 minutes

    # Calculate number of segments needed
    segments = len(audio) // length_ms
    print(f"Audio has {segments} segments")

    # Loop through segments
    for i in range(segments):
        # Calculate start and end times for this segment
        start_time = i * length_ms
        end_time = (i + 1) * length_ms

        # If it's the last segment (and there is a remainder), make sure to include it
        if i == segments - 1:
            end_time += len(audio) % length_ms

        # Extract segment
        segment = audio[start_time:end_time]

        # Save segment
        segment.export(f"{v_path}_{i}.mp3", format="mp3")
        print(f"Saved {v_path}_{i}.mp3")

# Step 4: Extract text from audio file
# OpenAI API key from environment variable
openai.api_key = os.environ["OPENAI_API_KEY"]

# Read audio file
print("Extracting text from audio")
txt = []
if segments == 0:
    file = open(f"{v_path}.mp3", "rb")
    transcription = openai.Audio.transcribe("whisper-1", file)
    txt.append(transcription["text"])
else:
    for i in range(segments):
        print(f"Extracting text from {v_path}_{i}.mp3")
        file = open(f"{v_path}_{i}.mp3", "rb")
        transcription = openai.Audio.transcribe("whisper-1", file)
        txt.append(transcription["text"])

# Step 5: Save text to a text file
with open(f"{v_path}.txt", "w") as f:
    for t in txt:
        f.write(t)

# Step 6: Delete audio files
print("Deleting audio files")
if segments > 0:
    for i in range(segments):
        os.remove(f"{v_path}_{i}.mp3")
