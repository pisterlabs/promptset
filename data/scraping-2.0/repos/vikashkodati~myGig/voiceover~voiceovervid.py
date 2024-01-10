import os
import base64
import pathlib
from pathlib import Path
import openai
from openai import OpenAI
import cv2
from moviepy.editor import VideoFileClip, AudioFileClip
import time

def open_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()
    
client = OpenAI(api_key=open_file('openai_api_key.txt'))

def extract_frames(video_path, output_folder, frame_interval=60):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return []

    frame_count = 0
    extracted_frame_paths = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = f'{output_folder}/frame_{frame_count}.jpg'
            cv2.imwrite(frame_filename, frame)
            extracted_frame_paths.append(frame_filename)
        
        frame_count += 1

    cap.release()
    print(f"Frame extraction complete. {len(extracted_frame_paths)} frames extracted.")
    return extracted_frame_paths
    

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return 0
    
    #calculate the total duration of the video in seconds
    return cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)

# Function to convert image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')
    
def get_frame_descriptions(frame_paths, finalprompt, max_retries=3, retry_delay=2):
    base64_frames = [image_to_base64(path) for path in frame_paths]

    prompt_messages = [
        {
            "role": "user",
            "content": [
                finalprompt,
                *map(lambda x: {"image": x, "resize": 768}, base64_frames)
            ]
        }
    ]

    params = {
        "model": "gpt-4-vision-preview",
        "messages": prompt_messages,
        "max_tokens": 1000,
        "temperature": 0
    }

    for attempt in range(max_retries):
        try:
            result = client.chat.completions.create(**params)
            return result.choices[0].message.content
        except openai.InternalServerError as e:
            print(f"Server error on attempt {attempt + 1}: {e}. Retrying after {retry_delay} seconds...")
            time.sleep(retry_delay)
    print("Failed to obtain descriptions after several retries.")
    return None


def create_voiceover(text, output_audio_path, model="tts-1", voice="echo"):
    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=text
    )
    response.stream_to_file(Path(output_audio_path))



def merge_audio_video(video_path, audio_path, output_video_path):
    video_clip = VideoFileClip(video_path)
    audio_clip = AudioFileClip(audio_path)

    final_clip = video_clip.set_audio(audio_clip)
    final_clip.write_videofile(output_video_path, codec="libx264", audio_codec="aac")


    # Main execution
video_path = '/Users/vikashkodati/code/myGig/voiceover/sample.mp4'
output_folder = 'extracted_frames'

# Calculate video duration and adjust prompt
video_duration = get_video_duration(video_path)
print(f"Video duration: {video_duration} seconds.")
word_count = video_duration * 2.5
print(f"Word Count:{word_count}")
prompt = f"(This video is ONLY {video_duration} seconds long, so make sure the voiceover MUST be less than {word_count} words)"
finalprompt = "ACT as a Tutorial Guide. In a conversational style explain in a Step-by-Step way what is happening in the frames suitable for a voiceover" + prompt

frame_paths = extract_frames(video_path, output_folder)

if frame_paths:
    descriptions = get_frame_descriptions(frame_paths, finalprompt)
    print("Descriptions obtained from GPT Vision API:")
    print(descriptions)

    # Create and save voiceover
    output_audio_path = 'voiceover.mp3'
    create_voiceover(descriptions, output_audio_path)

    # Merge audio with video and save as new file
    output_video_path = 'tt2.mp4'
    merge_audio_video(video_path, output_audio_path, output_video_path)
else:
    print("No frames were extracted.")