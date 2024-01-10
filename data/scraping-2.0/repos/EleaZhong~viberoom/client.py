
import os
import io
import requests
import pyaudio
import wave
from google.cloud import speech
from google.cloud import language_v2
from matplotlib import pyplot as plt
from queue import Queue
import logging
from dotenv import load_dotenv
import threading
from openai import OpenAI
import openai
import concurrent.futures
import shutil


load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def set_up():
    os.makedirs("logs", exist_ok=True)
    os.makedirs("recordings", exist_ok=True)
    os.makedirs("notes", exist_ok=True)


def setup_recording_logger():
    recording_logger = logging.getLogger("recording")
    recording_logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler("logs/record.log")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")
    file_handler.setFormatter(formatter)
    recording_logger.addHandler(file_handler)
    return recording_logger


def setup_processing_logger():
    processing_logger = logging.getLogger("processing")
    processing_logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler("logs/process.log")
    formatter = logging.Formatter("%(asctime)s [%(levelname)s]: %(message)s")
    file_handler.setFormatter(formatter)
    processing_logger.addHandler(file_handler)
    return processing_logger


def record_audio(filename):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 2

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    recording_logger.info("Recording...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)
    recording_logger.info("Finished recording")
    stream.stop_stream()
    stream.close()
    p.terminate()
    wf = wave.open(filename, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()



def transcribe_audio(filename):
    client = OpenAI()
    with open(filename, "rb") as f:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=f, language="en")
        print(transcript)
    return transcript.text


def analyze_sentiment(text_content: str = "I am so happy and joyful.") -> [float, float, list, list]:

    client = language_v2.LanguageServiceClient()

    # Define the type of document
    document_type = language_v2.Document.Type.PLAIN_TEXT

    # Set up the document with the content and type
    document = {
        "content": text_content,
        "type_": document_type,
        "language_code": "en",  # Language code (optional, auto-detects if not provided)
    }

    encoding_type = language_v2.EncodingType.UTF8

    response = client.analyze_sentiment(
        request={"document": document, "encoding_type": encoding_type}
    )

    # Extracting document sentiment
    document_score = response.document_sentiment.score
    document_magnitude = response.document_sentiment.magnitude

    # Extracting sentence level sentiments
    sentiments = [sentence.sentiment.score for sentence in response.sentences]
    magnitudes = [sentence.sentiment.magnitude for sentence in response.sentences]

    return document_score, document_magnitude, sentiments, magnitudes


def send_data_to_server(sentiment_data):
    sentiment_score, sentiment_magnitude = sentiment_data
    data = {
        'sentiment_score': sentiment_score,
        'sentiment_magnitude': sentiment_magnitude
    }
    print(data)
    response = requests.post('http://localhost:8000/api/v1/sentiment', json=data)

    if response.status_code == 200:
        print('Sentiment analysis data sent to server successfully.')
    else:
        print('Failed to send sentiment analysis data to server.', response.status_code)



def transcribe_and_analyze_and_send(filename):
    transcription = transcribe_audio(filename)
    document_score, document_magnitude, sentiments, magnitudes = analyze_sentiment(transcription)
    send_data_to_server((document_score, document_magnitude))

def main():
    recording_number = 0  # Make sure this starts as an integer
    last_filename = None

    if os.path.exists("recordings"):
        shutil.rmtree("recordings")
    os.makedirs("recordings")
    
    while True:
        # Convert recording_number to int to make sure the formatting works
        recording_number = int(recording_number)
        
        filename = f"recordings/recording-{recording_number:04d}.wav"
        record_audio(filename)
        recording_logger.info("Filename: " + str(filename))
        

        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.submit(transcribe_and_analyze_and_send, filename)
        
        recording_number += 1

if __name__ == '__main__':
    set_up()
    recording_logger = setup_recording_logger()
    processing_logger = setup_processing_logger()
    main()
