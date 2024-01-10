import openai
from docx import Document
from dotenv import load_dotenv, find_dotenv
from moviepy.video.io.VideoFileClip import VideoFileClip
import os
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage

load_dotenv(find_dotenv())
openai.api_key = os.environ["OPENAI_API_KEY"]

def extract_audio(input_video_path):
    input_video_path = input_video_path 

    # Path to save the extracted audio
    output_audio_path = 'output_audio.wav'

    # Load the video clip
    video_clip = VideoFileClip(input_video_path)

    # Extract audio from the video
    audio_clip = video_clip.audio

    # Save the audio to a file
    audio_clip.write_audiofile(output_audio_path)

    # Close the video and audio clips
    audio_clip.close()
    video_clip.close()

    # print("Audio extraction complete.")



def transcribe_audio(audio_file_path):
    with open(audio_file_path, 'rb') as audio_file:
        transcription = openai.Audio.transcribe("whisper-1", audio_file)
    return transcription['text']

# extract_audio(r"test\videoplayback.mp4")
text = transcribe_audio(r"test\test_audio.wav")

chat = ChatOpenAI(temperature=0.0, model_name="gpt-3.5-turbo-16k")

system_prompt = f"""
You are an AI model that takes the answers for an interview and gives the sentiment analysis of the overall answer and shows why the candidate may be a good fit or bad fit for the company. 
IMPORTANT do not reply with "As an AI model..." under any circumstances 
"""

human_message_example = """
    1. Could you tell me about yourself and describe your background in brief?
    
    I come from a small town, where opportunities were limited. Since good schools were a rarity, I started using online learning to stay up to date with the best. That’s where I learned to code and then I went on to get my certification as a computer programmer. After I got my first job as a front-end coder, I continued to invest time in mastering both front- and back-end languages, tools, and frameworks
"""

AI_message_example = """
    The answer of the candidate shows that he is aware about what he is talking and has a good knowledge of the field. An overall positive sentiment is shown by the answer, and we can say that the candidate is a good fit for the company.
"""

def func_(data):
    store = chat(
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_message_example),
            AIMessage(content=AI_message_example),
            HumanMessage(content=data),
        ]
    )
    return store

store = func_(data=text)

print(store.content)