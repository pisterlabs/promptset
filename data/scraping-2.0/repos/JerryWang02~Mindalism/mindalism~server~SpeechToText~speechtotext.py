from __future__ import division

import re
import sys

from google.cloud import speech

import pyaudio
from six.moves import queue

import os
import openai
from datetime import datetime
from datetime import date
import json

import nltk
nltk.downloader.download('vader_lexicon')

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm.notebook import tqdm
import requests

# API Keys
f = open("Mindalism/mindalism/server/SpeechToText/api.json")
data = json.load(f)
emotion_key = data["emotion"]
openai_key= data["openai"]
f.close()

# API Keys
f = open("api.json")
data = json.load(f)
emotion_key = data["emotion"]
openai_key= data["openai"]
f.close()

# Audio recording parameters
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

file = open('textfile.txt', 'w')

class MicrophoneStream(object):
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self, rate, chunk):
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(self, type, value, traceback):
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        """Continuously collect data from the audio stream, into the buffer."""
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)


def listen_print_loop(responses):
    """Iterates through server responses and prints them.
    The responses passed is a generator that will block until a response
    is provided by the server.
    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.
    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.
    """
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # The `results` list is consecutive. For streaming, we only care about
        # the first result being considered, since once it's `is_final`, it
        # moves on to considering the next utterance.
        result = response.results[0]
        if not result.alternatives:
            continue

        # Display the transcription of the top alternative.
        transcript = result.alternatives[0].transcript

        # Display interim results, but with a carriage return at the end of the
        # line, so subsequent lines will overwrite them.
        #
        # If the previous result was longer than this one, we need to print
        # some extra spaces to overwrite the previous result
        overwrite_chars = " " * (num_chars_printed - len(transcript))

        if not result.is_final:
            sys.stdout.write(transcript + overwrite_chars + "\r")
            sys.stdout.flush()

            num_chars_printed = len(transcript)

        else:
            print(transcript + overwrite_chars)
            file.write(transcript + overwrite_chars)

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r"\b(exit journal entry|quit)\b", transcript, re.I):
                print("Exiting..")
                break

            num_chars_printed = 0


def datacompliation():
    file.close()
    # Date
    _date = datetime.utcnow().isoformat()

    # Entry
    with open('textfile.txt') as infile:
        contents = infile.read()
        entry = contents
        print(contents)
    # os.remove('textfile.txt')

    # Summary
    openai.api_key = openai_key

    sia = SentimentIntensityAnalyzer()
    polarityScore = sia.polarity_scores(entry)


    emotion_prompt = "Classify the emotion in this post:"
    emotion_prompt += entry
    emotionResult = openai.Completion.create(
        model="text-davinci-003",
        prompt= emotion_prompt,
        temperature=0,
        max_tokens=30,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
    )
    emotionResult = emotionResult["choices"][0]["text"].replace("\n", "")

    response = openai.Image.create(
    prompt=emotionResult,
    n=1,
    size="1024x1024"
    )
    image_url = response['data'][0]['url']

    analytics = {
        "polarityScore": polarityScore,
        "emotionResult": emotionResult,
        "image": image_url
    }

    date_to_append = datetime.now().strftime("%Y-%m-%d")
    time_to_append = datetime.now().strftime("%H:%M:%S")

    f = open('JSON/journals.json')
    existing_data = json.load(f)

    prompt = "Summarize this journal entry for the reader and focus on any highlights or feelings that the writer was writing about for that day: "
    entry = entry.replace("Quit", "")
    entry = entry.replace("quit", "")
    
    if date_to_append in existing_data["dates"]:
        count = 0
        for time in existing_data["dates"][date_to_append].items():
            if time != "summary":
                count+=1

        count = count-1
        prompt += str(existing_data["dates"][date_to_append]["summary"]) + entry
        response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0.3, max_tokens=1000)
        complete_summary = response["choices"][0]["text"]
        complete_summary = complete_summary.replace("\n", "")


        data = {
            count : {
                "entry": entry,
                "analytics": analytics
            } 
        }
        existing_data["dates"][date_to_append].update(data)

        with open('JSON/journals.json', 'w') as f:
            json.dump(existing_data, f)
    else:
        prompt += entry

        response = openai.Completion.create(model="text-davinci-003", prompt=prompt, temperature=0.3, max_tokens=1000)
        complete_summary = response["choices"][0]["text"]
        complete_summary = complete_summary.replace("\n", "")

        data = {
            date_to_append : {
                "summary": complete_summary,
                0 : {
                    "entry": entry,
                    "analytics": analytics
                },
            }, 
        }
    existing_data["dates"].update(data)
    # Json File

    filename2 = "../../client/src/journals.json"
    filename =  "JSON/journals.json"
    json_data = json.dumps(data)
    with open(filename, "w") as outfile:
        outfile.write(json_data)

    with open(filename2, "w") as outfile:
        outfile.write(json_data)
        
    print(data)

def main():
    # See http://g.co/cloud/speech/docs/languages
    # for a list of supported languages.
    language_code = "en-US"  # a BCP-47 language tag

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
    )

    streaming_config = speech.StreamingRecognitionConfig(
        config=config, interim_results=True
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        audio_generator = stream.generator()
        requests = (
            speech.StreamingRecognizeRequest(audio_content=content)
            for content in audio_generator
        )

        responses = client.streaming_recognize(streaming_config, requests)

        # Now, put the transcription responses to use.
        listen_print_loop(responses)
        datacompliation()

if __name__ == "__main__":
    main()

# file.close()