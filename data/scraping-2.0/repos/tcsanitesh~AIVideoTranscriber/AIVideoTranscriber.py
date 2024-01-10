from sre_constants import SUCCESS
#from tkinter.ttk import Progressbar
import streamlit as st
from moviepy.video.io.VideoFileClip import VideoFileClip
import pandas as pd
import os
#from dotenv import load_dotenv
from azure.cognitiveservices.speech import SpeechConfig, SpeechRecognizer, AudioConfig
from azure.cognitiveservices.speech import ResultReason
import openai
from datetime import datetime
#import time

# Load environment variables from the .env file
#load_dotenv()
def AIVT_services(root_folder,file_path,OPENAI_API_Key,Speech_API_KEY1):

  # # Function to check if the file is an audio file
  # def is_audio_file(file):
  #   return file.lower().endswith(".wav")
   
  # Function to check if the file is a video file
  def is_video_file(file):
    return file.lower().endswith(".mp4")
   
   
  # Function to transcribe audio from video file using Speech-to-Text API
  def transcribe_audio(audio_file):
     
    speech_key = Speech_API_KEY1
    service_region = "eastus"
     
    # Set up the Speech-to-Text API
    speech_config = SpeechConfig(subscription=speech_key, region=service_region)
    audio_config = AudioConfig(filename=audio_file)
   
    # Initialize the SpeechRecognizer
    speech_recognizer = SpeechRecognizer(speech_config=speech_config, audio_config=audio_config)
   
    # Perform speech recognition
    result = speech_recognizer.recognize_once()
   
    # Get the transcript summary
    if result.reason == ResultReason.RecognizedSpeech:
      transcript_summary = result.text 
    else:
      transcript_summary = "Transcription failed or no speech recognized."
   
    return transcript_summary
   
   
  # Function to generate the title, description, keywords, and category using OpenAI API
  def generate_summary(input_transcript):
   
    openai.api_key = OPENAI_API_Key 
     
    response = openai.Completion.create(
      engine="text-davinci-003",
      prompt="Transcript Summary: " + input_transcript + "\nHelp me Generate a \nTitle:, \n\nDescription:{summarization of transcript}, \nKeywords :, and \nCategory: for a promotional material based on the given transcript:\nNote: The reference of \"Tiki Well\" is actually \"TQL\" if any which is a logistic company.",
      max_tokens=1000,
      temperature=0
    )
     
    # Extracting the generated text
    generated_text = response.choices[0].text.strip()
     
    # Splitting the generated text into lines
    lines = generated_text.split('\n')
   
    title = lines[0]
    description = lines[1]
    keywords = lines[2]
    category = lines[3]
   
   
    return title, description, keywords, category

   
  file_metadata = []
  file_type = root_folder.type
  file_name = root_folder.name

   

  if is_video_file(file_name):
    video_clip = VideoFileClip(file_path)
    audio_file_path = os.path.splitext(file_path)[0] + ".wav"
    #st.write(audio_file_path)
    audio_clip = video_clip.audio

    #transcript_summary = transcribe_audio(audio_file_path)
    audio_clip.write_audiofile(audio_file_path, codec='pcm_s16le')
    audio_clip.close()
    video_clip.close()
  else:
    audio_file_path = file_path

  transcript_summary = transcribe_audio(audio_file_path)
  #file_size = f"{os.path.getsize(file_path)/ (1024 * 1024):.2f}"
  file_size=root_folder.size/1024/1024 #convert to MB
  # Generate title, description, keywords, and category using OpenAI API
  if transcript_summary != "Transcription failed or no speech recognized.":
    title, description, keywords, category = generate_summary(transcript_summary)
   
  else :
    title, description, keywords, category = "NA", "Transcription failed or no speech recognized to extract Title, Description, Keywords, and Category.", "NA", "NA"
   
  # Get the creation date of the video file from the metadata
  #creation_date = datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d')
   
  # Get the current date as the modified date
  modified_date = datetime.now().strftime('%Y-%m-%d')
   
  # Append the metadata and transcript summary to the list
  file_metadata.append({
    "File Path": file_path,
    "File Name": file_name,
    "File Size (in MB)": file_size,
    "Transcript Summary": transcript_summary,
    "Title": title,
    "Description": description,
    "Keywords": keywords,
    "Category": category,
    "Modified ON": modified_date
    # Add other extracted metadata here
    })
  c1,c2,c3=st.columns([1,1,1])
  c1.write(title)
  c1.write(description)
  c2.video(root_folder)
  c1.write(category)
  c1.write(keywords)
  c3.markdown("""#### Transcript\n """+ transcript_summary)

  return file_metadata