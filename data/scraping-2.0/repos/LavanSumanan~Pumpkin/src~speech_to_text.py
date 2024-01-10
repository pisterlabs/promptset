from dotenv import load_dotenv
import os
import openai

# testing
from time import perf_counter

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def transcribe(filename):
    audio_file = open(filename, "rb")
    transcript = openai.Audio.translate(model="whisper-1", file=audio_file, response_format="text")
    return transcript

def transcribe_multiprocessing(jobs, results):
    while True:
        job_input = jobs.get()
        if not job_input:
            print("ending processing")
            break
        (index, filename) = job_input
        t_start = perf_counter()
        audio_file = open(filename, "rb")
        transcript = openai.Audio.translate(model="whisper-1", file=audio_file, response_format="text")
        t_stop = perf_counter()
        print("Time taken to transcribe (seconds): ", round(t_stop-t_start, 2))
        results.put((index, transcript))
