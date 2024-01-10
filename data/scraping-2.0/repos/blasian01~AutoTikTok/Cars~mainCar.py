import openai
import random 
import os
import requests
import json
import io
import base64
from PIL import Image, PngImagePlugin
from moviepy.editor import *
from api import *
import moviepy.editor as mp
import logging
import subprocess
from datetime import datetime

#----------------------------------------------------------------------------------------------------

# Replacing prompt with new deform prompt from GPT
def replace_prompts(target_file, source_file):
    # Load the content from the source file
    with open(source_file, 'r', encoding='utf-8') as file:
        source_content = json.load(file)
    
    # Load the content from the target file
    with open(target_file, 'r', encoding='utf-8') as file:
        target_content = json.load(file)
    
    # Replace the 'prompts' section in the target content with the source content
    target_content['prompts'] = source_content
    
    # Write the updated content back to the target file
    with open(target_file, 'w', encoding='utf-8') as file:
        json.dump(target_content, file, indent=4)
        
#-----------------------------------------------------------------------------------------------

# Functions for creating video with sound
def get_latest_file(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    files.sort(key=lambda x: os.path.getmtime(os.path.join(folder_path, x)), reverse=True)
    return os.path.join(folder_path, files[0]) if files else None

def get_random_file(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return os.path.join(folder_path, random.choice(files)) if files else None

def add_audio_to_video(video_path, audio_path, output_path):
    try:
        video = VideoFileClip(video_path)
        audio = AudioFileClip(audio_path)
    except Exception as e:
        print(f"Error loading media files: {e}")
        return

    # Handling different durations between video and audio
    if video.duration < audio.duration:
        # Looping the video to match the audio duration
        video = video.fx(vfx.loop, duration=audio.duration)
    elif video.duration > audio.duration:
        # Shortening the video to match the audio duration
        video = video.subclip(0, audio.duration)

    video_with_audio = video.set_audio(audio)

    # Ensure the output_path ends with .mp4
    if not output_path.endswith('.mp4'):
        output_path += '.mp4'

    try:
        video_with_audio.write_videofile(output_path, codec="libx264")
        print(f"Successfully combined {video_path} with {audio_path}. Output saved to {output_path}.")
    except Exception as e:
        print(f"Error writing video file: {e}")

     
#----------------------------------------------------------------------------------------------------

if __name__ == "__main__":

    #Making Diffusion Prompt for the car that was selected
    #subprocess.run(["python", "prompt.py"])
    with open("output_folder/prompt.txt") as f:
        generated_prompt = f.read()
    generated_prompt = generated_prompt + ","

    
    # Setting the path for deforum settings
    settings_file_path = 'deforum_settings.txt'
    promptFile = 'output_folder/prompt.txt'

    # Call the function to replace the prompts
    replace_prompts(settings_file_path, promptFile) # Failing to properly input the new prompt
    
    job_id = send_post_request('deforum_settings.txt', 'output_folder/RawVideo') 
    if job_id:
        print(f"The job_id is: {job_id}")
        output_directory = get_output_directory(job_id)
        print(f"The output directory is: {output_directory}")
        if output_directory:
            print(f"The output directory is: {output_directory}")
            copy_video_to_output_folder(output_directory, 'output_folder/RawVideo') # not pulling correct video file path 
    else:
        print("Failed to get the job_id.")
    
    # File locations for video and audio
    video_folder = "output_folder/RawVideo"
    audio_folder = "output_folder/Music"
    
    # Get the current date and format it as YYYYMMDD
    current_date = datetime.now().strftime('%Y%m%d')
    
    # Setting output file path and name for the tiktok
    output_file_name = f"output_video_{current_date}.mp4"  # Appending the date to the file name
    output_path = f"output_folder/FinishedTikToks/{output_file_name}"  # Full path to the output file

    # Getting latest video and a random sound to pair with
    latest_video = get_latest_file(video_folder)
    random_audio = get_random_file(audio_folder)

    # If there is video and audio that go together then combine them to make the tiktok
    if latest_video and random_audio:
        add_audio_to_video(latest_video, random_audio, output_path)
        print(f"Successfully combined {latest_video} with {random_audio}. Output saved to {output_path}.")
    else:
        print("Couldn't find files to process.")
