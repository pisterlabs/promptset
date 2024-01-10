
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
import json


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
    RECORD_SECONDS = 3

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


def continuous_recording(queue):
    recording_number = 0  # Make sure this starts as an integer
    last_filename = None
    
    while True:
        # Convert recording_number to int to make sure the formatting works
        recording_number = int(recording_number)
        
        filename = f"recordings/recording-{recording_number:04d}.wav"
        record_audio(filename)
        recording_logger.info("Filename: " + str(filename))
        
        # if last_filename:
        #     combined_filename = f"recordings/combined-{recording_number:04d}.wav"
            
        #     with wave.open(combined_filename, 'wb') as outfile:
        #         # Open first file to read its headers
        #         with wave.open(last_filename, 'rb') as infile:
        #             params = infile.getparams()
        #             outfile.setparams(params)
                
        #         # Now proceed to read frames from each file and write
        #         for fname in [last_filename, filename]:
        #             with wave.open(fname, 'rb') as infile:
        #                 outfile.writeframes(infile.readframes(infile.getnframes()))
            
        #     queue.put(combined_filename)
        #     recording_logger.info("Combined Filename: " + str(combined_filename))
        # else:
        queue.put(filename)
        
        last_filename = filename
        recording_number += 1


def transcribe_audio(filename):
    client = OpenAI()
    with open(filename, "rb") as f:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=f, language="en")
    return transcript["text"]

def analyze_sentiment(text_content: str = "I am so happy and joyful."):
    """
    Analyzes Sentiment in a string using OpenAI's ChatGPT.

    Args:
      text_content: The text content to analyze.
    """

    client = OpenAI()

    response = client.chat.completions.create(
        model="gpt-3.5-turbo-1106",
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": "Analyze the sentiment of the provided transcript and return sentiment analysis as a JSON object. The sentiment_score should range from 0 to 10, where 0 represents extremely negative sentiment and 10 represents extremely positive sentiment. The intensity_score should range from 0 to 10, where 0 represents extremely subdued emotions and 10 represents extremely intense emotions. The agreement_score from 0 to 10, where 0 represents extremely low agreement between statements/sentences/conversation and 10 represents extremely high agreement between statements/sentences/conversation. The scores should be a floating-point number. \n User Input:\"The service at the restaurant was outstanding and the staff was incredibly friendly! Yea I know right it's so great. I'm totally in love with it\" \n Expected System Output: \n {\"sentiment_score\": 9.2, \"intensity_score\": 7.9, \"agreement_score\": 8.5}"},
            {"role": "user", "content": text_content}
        ]
    )


    raw_sentiment_score = response.choices[0].message.content
    sentiment_score = json.loads(raw_sentiment_score)["sentiment_score"]
    intensity = json.loads(raw_sentiment_score)["intensity_score"]
    agreement = json.loads(raw_sentiment_score)["agreement_score"]
    print(f"Sentiment score: {sentiment_score}")
    return sentiment_score, intensity, agreement

def send_to_server(sentiment_score, sentiment_magnitude):
    """
    Send the sentiment analysis data to the server
    """
    data = {
        'sentiment_score': sentiment_score,
        'sentiment_magnitude': sentiment_magnitude
    }
    response = requests.post('http://localhost:8000/api/v1/sentiment', json=data)

    return response.status_code

def process_audio(queue):
    last_sentences = []
    while True:
        processing_logger.info(f"Queue Length: {queue.qsize()}")
        filename = queue.get()
        transcript = transcribe_audio(filename)
        last_sentences.append(transcript)
        sentiments, magnitudes = analyze_sentiment(" ".join(last_sentences[-3:]))
        processing_logger.info("Filename: " + str(filename))
        processing_logger.info("Transcript: " + str(transcript))
        processing_logger.info("Document Score: " + str(document_score))
        processing_logger.info("Document Magnitude: " + str(document_magnitude))
        processing_logger.info("Sentiments: " + str(sentiments))
        processing_logger.info("Magnitudes: " + str(magnitudes))
        status_code = send_to_server(document_score, document_magnitude)
        if status_code == 200:
            processing_logger.info('Sentiment analysis data sent to server successfully.')
        else:
            processing_logger.info('Failed to send sentiment analysis data to server.')


def main():

    recording_queue = Queue()

    recording_thread = threading.Thread(
        target=continuous_recording, args=(recording_queue,))
    recording_thread.start()

    processing_thread = threading.Thread(
        target=process_audio, args=(recording_queue,))
    processing_thread.start()

if __name__ == '__main__':
    # set_up()
    # recording_logger = setup_recording_logger()
    # processing_logger = setup_processing_logger()
    # main()
    print(analyze_sentiment("Fuck santa! yea i totally don't understand the um appeal of a old white man. fuck him. fuck. yea. um."))
