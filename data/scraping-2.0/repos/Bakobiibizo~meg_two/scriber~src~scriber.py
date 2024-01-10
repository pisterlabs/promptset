import os
from moviepy.editor import AudioFileClip
from pydub import AudioSegment
import datetime
import pretty_notes
import openai


def process_video_file(file_path):
    """
    Extracts audio from the video file using moviepy.

    :param file_path: The path of the video file.
    :return: The path of the extracted audio file.
    """
    video = AudioFileClip(file_path)
    audio_file_path = file_path + ".wav"
    video.write_audiofile(audio_file_path)
    return audio_file_path


def convert_audio_format(audio_file_path):
    """
    Converts the audio file to the desired format and sample rate.

    :param audio_file_path: The path of the audio file.
    :return: The path of the converted audio file.
    """
    audio = AudioSegment.from_file(audio_file_path)
    audio = audio.set_frame_rate(16000)
    converted_file_path = audio_file_path.replace(".wav", ".mp3")
    audio.export(converted_file_path, format="mp3")
    return converted_file_path


def transcribe_audio(audio_file_path):
    """
    Transcribes the audio file using OpenAI.

    :param audio_file_path: The path of the audio file.
    :return: The transcription text.
    """
    with open(audio_file_path, 'rb') as audio_file:
        transcription = openai.Audio.transcribe("whisper-1", audio_file)
        return transcription['text']


def extract_abstract_summary(transcription):
    """
    Extracts the abstract summary from the transcription using OpenAI.

    :param transcription: The transcription text.
    :return: The abstract summary.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a highly skilled AI trained in language comprehension and summarization. I would like you to read the following text and summarize it into a concise abstract paragraph. Aim to retain the most important points, providing a coherent and readable summary that could help a person understand the main points of the discussion without needing to read the entire text. Please avoid unnecessary details or tangential points."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']


def extract_key_points(transcription):
    """
    Extracts the key points from the transcription using OpenAI.

    :param transcription: The transcription text.
    :return: The key points.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a proficient AI with a specialty in distilling information into key points. Based on the following text, identify and list the main points that were discussed or brought up. These should be the most important ideas, findings, or topics that are crucial to the essence of the discussion. Your goal is to provide a list that someone could read to quickly understand what was talked about."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']


def extract_action_items(transcription):
    """
    Extracts the action items from the transcription using OpenAI.

    :param transcription: The transcription text.
    :return: The action items.
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are an AI expert in analyzing conversations and extracting action items. Please review the text and identify any tasks, assignments, or actions that were agreed upon or mentioned as needing to be done. These could be tasks assigned to specific individuals, or general actions that the group has decided to take. Please list these action items clearly and concisely."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']


def abstract_summary_extraction(transcription):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "You are a highly skilled AI trained in language comprehension and summarization. I would like you to read the following text and summarize it into a concise abstract paragraph. Aim to retain the most important points, providing a coherent and readable summary that could help a person understand the main points of the discussion without needing to read the entire text. Please avoid unnecessary details or tangential points."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']

def sentiment_analysis(transcription):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": "As an AI with expertise in language and emotion analysis, your task is to analyze the sentiment of the following text. Please consider the overall tone of the discussion, the emotion conveyed by the language used, and the context in which words and phrases are used. Indicate whether the sentiment is generally positive, negative, or neutral, and provide brief explanations for your analysis where possible."
            },
            {
                "role": "user",
                "content": transcription
            }
        ]
    )
    return response['choices'][0]['message']['content']

def save_as_file(minutes):
    human_readable_text = ''
    for key, value in minutes.items():
        human_readable_text += f'{key}: {value}\n'

    with open('out/minutes.txt', 'w') as file:
        file.write(human_readable_text)

transcription = None
transcription = transcribe_audio(audio_file_name)
minutes = None
if transcription:
    minutes = meeting_minutes(transcription)
    print(minutes)
    save_as_file(minutes)
    pretty_notes.save()
else:
    print("No transcription available.")


