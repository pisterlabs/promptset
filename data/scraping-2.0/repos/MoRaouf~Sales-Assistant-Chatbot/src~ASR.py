# Automatic Speech Recognition

import openai
from audio_recorder_streamlit import audio_recorder



def record_audio():
    """Initializa audio recorder & record audio"""
    
    return audio_recorder()



def save_audio(AUDIO, TEMP_AUDIO_PATH):
    """Save audio to a file

    Args:
        audio (str): Transctiption text to be saved
        temp_audio_path (str): Path of file to save transcription
    """
     
    with open(TEMP_AUDIO_PATH, "wb") as f:
        f.write(AUDIO)
     


def transcribe_audio(TEMP_AUDIO_PATH, api_key:str = None):
    """Transcribe an audio file using OpenAI Whisper API

    Args:
        audio_file_path (str): Path to audio file

    Returns:
        response: textual transcription of audio
    """
    try:
        with open(TEMP_AUDIO_PATH, "rb") as audio_file:
            response = openai.Audio.transcribe("whisper-1", audio_file, api_key=api_key)
        return response["text"]
    
    except Exception as e:
        print(f"Error calling Whisper API: {str(e)}")
        return None
    

def save_transcription(TRANSCRIPTION, TRANSCRIPTION_PATH):
    """Save transcription to a file

    Args:
        transcription (str): Transctiption text to be saved
        transciption_path (str): Path of file to save transcription
    """

    with open(TRANSCRIPTION_PATH, "w+") as f:
            f.write(TRANSCRIPTION)
