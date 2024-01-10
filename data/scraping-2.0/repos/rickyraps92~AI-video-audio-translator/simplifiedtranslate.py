import os
import openai
from moviepy.editor import VideoFileClip

# OpenAI API configuration
openai.api_key = "YOUR_API_KEY"

def process_video(video_path):
    # Extract audio from the video
    audio_path = "Download_audio.wav"
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audio_path)

    # Open the audio file
    audio_file = open(audio_path, "rb")

    # Translate the audio using OpenAI's Whisper API
    response = openai.Audio.translate("whisper-1", audio_file)

    # Close the audio file
    audio_file.close()

    # Get the translated text
    translated_text = response['text']

    # Write the translated text to a text file
    output_text_file = os.path.splitext(video_path)[0] + "_translated.txt"
    with open(output_text_file, 'w') as file:
        file.write(translated_text)

    print(f"Translation written to {output_text_file}")

# Get all video files in the current directory
video_files = [file for file in os.listdir() if file.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

# Process each video file
for video_file in video_files:
    process_video(video_file)
