from pytube import YouTube
from moviepy.editor import *
from moviepy.video.io.VideoFileClip import VideoFileClip
import math
from pydub import AudioSegment
from flask import current_app
from watchtower import CloudWatchLogHandler
import os
import uuid
import openai
import requests
from datetime import datetime, timedelta
import asyncio
import aiofiles
import aiohttp
import os
import tempfile
from pydub.utils import mediainfo
import numpy as np
import multiprocessing
from functools import partial


OPEN_AI_KEY = "''"
openai.api_key = OPEN_AI_KEY
WHISPER_MODEL_NAME = 'whisper-1'
CHUNKS_SIZE = 25 * 60 * 1000  # 25 minutes in milliseconds



def batch_transcribe_file(model_id, path):
    # Split audio file into chunks
    audio = AudioSegment.from_file(path)
    segments = []
    for i in range(0, len(audio), CHUNKS_SIZE):
        segment = audio[i:i+CHUNKS_SIZE]
        segments.append(segment)

    pool = multiprocessing.Pool()
    transcribe_func = partial(transcribe_file, model_id)
    transcripts = pool.map(transcribe_func, [(i, segment, path) for i, segment in enumerate(segments)])

    return transcripts  
    

def transcribe_file(model_id, segment_info):
    i, segment, path = segment_info
    segment_path = f"{path}_{i}.mp3"
    segment.export(segment_path, format="mp3", tags={"timecode": str(i*CHUNKS_SIZE)})
    url = 'https://api.openai.com/v1/audio/translations'
    headers = {'Authorization': f'Bearer {OPEN_AI_KEY}'}
    data = {'model': 'whisper-1',}
    print(segment_path)
    files = {
        'file': open(segment_path, 'rb'),
        'model': (None, 'whisper-1'),
        'response_format': (None, 'srt')
    }
    response = requests.post(url, headers=headers, files=files)
    if response.ok:
        return response.text
    else:
        # import pdb
        # pdb.set_trace()
        print(response.content.decode())
        raise ValueError(f"Request failed with status code {response.status_code}.")

def download_video(vidLink):
    try:
        vidObj = YouTube(vidLink)
        vidObj.check_availability()
    except Exception as e:
        print(e)
        return "vid not available"
    vidStreams = vidObj.streams.filter(only_audio=True)[0]
    if not vidStreams:
        return "no streams availables"   
    file_name = uuid.uuid4().__str__()
    outFile = vidStreams.download(output_path='audio')
    base, ext = os.path.splitext(outFile)
    newFile = file_name + '.mp3'
    os.rename(outFile, newFile)
    return newFile

def get_video_transcript(url, isMP4):
    '''for faster testing'''
    videoFile = url
    '''end for faster testing'''
    # print('#Downloading Video')
    # videoFile = ''
    # if isMP4:
    #     # download video from s3
    #     # videoFile = get_video_file(bucket,key)

    #     # convert to s3
    #     videoFileMP3 = f'{url}.mp3'
    #     video = VideoFileClip(url)
    #     audio = video.audio
    #     audio.write_audiofile(videoFileMP3)
    #     video.close()
    #     audio.close()

    #     # delete mp4
    #     # os.remove(url)
    #     videoFile = videoFileMP3
    # else:
    #     videoFile = download_video(url)
    # print('#Video Downloaded')
    transcripts =  batch_transcribe_file(WHISPER_MODEL_NAME, videoFile)
    to_save = np.array(transcripts)
    np.save('batched_transcripts.npy', to_save)
    # transcripts = transcribe_file(WHISPER_MODEL_NAME, videoFile)
    print(transcripts)
    # os.remove(videoFile)
    formatted_subtitles = srt_to_array(transcripts)
    print('#Transcripts Generated')
    return formatted_subtitles

def srt_to_array(array_of_srt_text):
    # Split the SRT text into an array of subtitles
    subtitles = []
    for i, srt_text in enumerate(array_of_srt_text):
        srt_array = srt_text.strip().split('\n\n')

        for s in srt_array:
            # Split each subtitle into its timecodes and text
            s_parts = s.split('\n')
            print(s_parts)
            # Extract start and end timecodes and convert to datetime objects
            start_time = datetime.strptime(s_parts[1].split(' --> ')[0], '%H:%M:%S,%f')
            end_time = datetime.strptime(s_parts[1].split(' --> ')[1], '%H:%M:%S,%f')

            # Add offset to start and end times:
            start_time += timedelta(milliseconds=CHUNKS_SIZE*i)
            end_time += timedelta(milliseconds=CHUNKS_SIZE*i)

            # Calculate start and end times in seconds
            start_time_seconds = (start_time - datetime(1900, 1, 1)).total_seconds()
            end_time_seconds = (end_time - datetime(1900, 1, 1)).total_seconds()
            # Create a dictionary object with start and end times in seconds and text
            subtitle = {'start': start_time_seconds, 'end': end_time_seconds, 'text': s_parts[2]}
            subtitles.append(subtitle)
    to_save = np.array(subtitles)
    np.save('subtitles.npy', to_save)
    return subtitles

if __name__=="__main__":
    # async def main():
    path = 'lex_short.mp3'
    subtitles =  get_video_transcript(path, True)
    print(subtitles)
    print(type(subtitles))
    # main()
