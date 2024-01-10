import pytube
from moviepy.editor import *
import os
import whisper
import torch
import openai
from dotenv import load_dotenv
from multiprocessing import Process

def convert_video_to_audio(url, output_path, max_duration):
    # Download the video from Google Drive or Youtube
    youtube = pytube.YouTube(url)
    video = youtube.streams.first()
    video.download()
    

    # Check video duration
    if youtube.length > max_duration * 3600:
        print("Video duration exceeds the maximum limit.")
        os.remove(video.default_filename)
        return

    # Extract the audio from the video
    video_file = video.default_filename
    video_clip = VideoFileClip(video_file)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(output_path)

    # Clean up temporary files
    audio_clip.close()
    video_clip.close()
    os.remove(video_file)



def audio_transcriber(audio_path):
    model_path = "model/base.pt"
    model = whisper.load_model(model_path)
    result_path = audio_path.replace(".wav",".txt")


    

    device = "cuda" if torch.cuda.is_available() else "cpu"  # Check if GPU is available, else use CPU
    model.to(device)

    result = model.transcribe(audio_path,fp16=False)
    print(result["text"])
    os.remove(audio_path)
    with open(result_path,'w') as file:
        file.write(result["text"])

def LLMSummarizer(video_url,prompt_path):
    openai.api_key = os.getenv('OPEN_AI_KEY')
    with open(prompt_path,'r') as prompt_file:
        prompt = prompt_file.read()
    messages = []
    print(prompt)
    messages.append({"role":"user","content":
                     f"Please summarize this video, the link of the video is {video_url} and the text content in the video is as follows: {prompt}"})

    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=messages
    )
    reply = response["choices"][0]["message"]["content"]
    os.remove(prompt_path)
    return reply



# convert_video_to_audio("https://youtu.be/PNTCM7cbrsc","output.wav",2)


