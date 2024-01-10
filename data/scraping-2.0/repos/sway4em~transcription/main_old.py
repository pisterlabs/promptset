import asyncio
import boto3
import os
from openai import OpenAI
from io import BytesIO
import requests
import sounddevice
import simpleaudio as sa
from pydub import AudioSegment
import time
from amazon_transcribe.client import TranscribeStreamingClient
from amazon_transcribe.handlers import TranscriptResultStreamHandler
from amazon_transcribe.model import TranscriptEvent
import amazon_transcribe
import json
from moviepy.editor import *
from moviepy.editor import VideoFileClip

ENABLE_STREAMING = False
synthesis_time = 0
API_URL = "https://api-inference.huggingface.co/models/dslim/bert-base-NER"
headers = {"Authorization":f"Bearer {os.environ['HF_KEY']}"}
is_playing_audio = False

def ner(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

client = OpenAI(
    # This is the default and can be omitted
    
    api_key=os.environ["OPENAI_API_KEY"]
)
polly_client = boto3.client('polly', region_name='us-west-2')

def get_response(message):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": message,
            }
        ],
        model="gpt-3.5-turbo",
    )
    return chat_completion.choices[0].message.content

def synthesize_speech(text, output_format='mp3', voice='Brian', enable_streaming=False, file_path='response.mp3'):
    start_time = time.time()  # Start the timer

    polly_client = boto3.client('polly', region_name='us-west-2')

    response = polly_client.synthesize_speech(VoiceId=voice,
                                              OutputFormat=output_format,
                                              Text=text)

    end_time = time.time() - start_time  # Measure the time taken
    print(f"Time taken to synthesize speech: {end_time - start_time} seconds")

    if enable_streaming:
        if "AudioStream" in response:
            return response['AudioStream'], file_path
        else:
            raise Exception("Could not stream audio from Polly response.")
    else:
        if "AudioStream" in response:
            audio_data = response['AudioStream'].read()

            # Save the audio data to a file
            save_audio_to_file(audio_data, file_path)

            return audio_data, file_path
        else:
            raise Exception("Could not synthesize audio from Polly response.")

def save_audio_to_file(audio_data, file_path='response.mp3'):
    """
    Save the audio data to a file.
    """
    with open(file_path, 'wb') as file:
        file.write(audio_data)
    print(f"Audio saved to {file_path}")

def play_audio(file_path, enable_streaming=False):
    global is_playing_audio
    is_playing_audio = True
    print("Playing audio...")
    start_time = time.time()  # Start the timer
    try:
        audio_segment = AudioSegment.from_file(file_path, format='mp3')
    except Exception as e:
        print("Error converting audio file:", e)
        is_playing_audio = False
        return

    try:
        play_obj = sa.play_buffer(
            audio_segment.raw_data,
            num_channels=audio_segment.channels,
            bytes_per_sample=audio_segment.sample_width,
            sample_rate=audio_segment.frame_rate
        )
        play_obj.wait_done()
    except Exception as e:
        print("Error playing audio:", e)
    finally:
        is_playing_audio = False
        end_time = time.time() - start_time
        print(f"Time taken to play audio: {end_time} seconds")

message = ""
                
class MyEventHandler(TranscriptResultStreamHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_transcript = ""
        self.partial_transcript = ""
        self.last_final_time = None
        self.silence_threshold = 3  # seconds
        self.person_name = None  # To store the identified person's name

    async def handle_transcript_event(self, transcript_event: TranscriptEvent):
        now = asyncio.get_event_loop().time()
        results = transcript_event.transcript.results
        for result in results:
            if result.is_partial:
                self.partial_transcript = ' '.join([alt.transcript for alt in result.alternatives])
                print("Partial:", self.partial_transcript)
            else:
                self.current_transcript = ' '.join([alt.transcript for alt in result.alternatives])
                print("Final:", self.current_transcript)
                self.last_final_time = now

                # Check for a person's name if not already found
                if not self.person_name:
                    self.check_for_name(self.current_transcript)

        if self.last_final_time and (now - self.last_final_time) > 4:
            print("Speech considered complete after silence.")
            response = get_response(self.current_transcript)
            print("Response:", response)
            print("Synthesizing response...")
            start_time = time.time()  # Start the timer
            _, audio_file_path = synthesize_speech(response, enable_streaming=ENABLE_STREAMING)
            end_time = time.time() - start_time  # Measure the time taken
            print(f"Time taken to synthesize speech: {end_time - start_time} seconds")
            # Play audio from the saved file
            print("Playing audio response...")
            play_audio(audio_file_path, enable_streaming=ENABLE_STREAMING)

            # Call Gooey API with the saved audio and predefined image URL
            print("Calling Gooey API...")
            start_time = time.time()  # Start the timer
            call_gooey_api(audio_file_path, "https://storage.googleapis.com/dara-c1b52.appspot.com/daras_ai/media/8c1b1f02-5f66-11ed-a8a9-02420a0000aa/ezgif-5-4ccc215641.gif")
            end_time = time.time() - start_time  # Measure the time taken
            print(f"Time taken to finish Gooey actions: {end_time - start_time} seconds")
            self.current_transcript = ""
            self.partial_transcript = ""
            self.last_final_time = None

    def check_for_name(self, text):
        print("Checking for name...")
        output = ner({"inputs": text})
        for out in output:
            if out["entity_group"] == "PER":
                self.person_name = out["word"]
                print(f"The person's name is {self.person_name}")
                break  # Stop after identifying the name
        print("Name check complete")
        print(output)

def call_gooey_api(audio_file_path, image_url):
    print("Preparing to call Gooey API...")
    global is_playing_audio
    is_playing_audio = True
    start_time = time.time()  # Start the timer
    with open(audio_file_path, "rb") as audio_file:
        files = [
            ("input_face", (image_url.split("/")[-1], requests.get(image_url).content)),
            ("input_audio", (audio_file_path, audio_file)),
        ]
        payload = {}

        response = requests.post(
            "https://api.gooey.ai/v2/Lipsync/form/",
            headers={"Authorization": "Bearer " + os.environ["GOOEY_API_KEY"]},
            files=files,
            data={"json": json.dumps(payload)},
        )

    assert response.ok, response.content
    result = response.json()
    end_time = time.time() - start_time  # Measure the time taken
    print(f"Time taken to call Gooey API: {end_time - start_time} seconds")
    # Download the video
    print("Downloading video...")
    start_time = time.time()  # Start the timer
    video_url = result["output"]["output_video"]
    video_response = requests.get(video_url)
    with open('output_video.mp4', 'wb') as video_file:
        video_file.write(video_response.content)
        print("Video downloaded as 'output_video.mp4'")

    print("Gooey API Response:", response.status_code, result)
    end_time = time.time() - start_time  # Measure the time taken
    print(f"Time taken to download video: {end_time - start_time} seconds")
    print("Playing downloaded video...")
    play_video_moviepy("output_video.mp4")
    is_playing_audio = False
    return result


async def mic_stream():
    # This function wraps the raw input stream from the microphone forwarding
    # the blocks to an asyncio.Queue.
    loop = asyncio.get_event_loop()
    input_queue = asyncio.Queue()

    def callback(indata, frame_count, time_info, status):
        if not is_playing_audio:  # Check if audio is being played back
            loop.call_soon_threadsafe(input_queue.put_nowait, (bytes(indata), status))


    # Be sure to use the correct parameters for the audio stream that matches
    # the audio formats described for the source language you'll be using:
    # https://docs.aws.amazon.com/transcribe/latest/dg/streaming.html
    stream = sounddevice.RawInputStream(
        channels=1,
        samplerate=16000,
        callback=callback,
        blocksize=1024 * 2,
        dtype="int16",
    )
    # Initiate the audio stream and asynchronously yield the audio chunks
    # as they become available.
    with stream:
        while True:
            indata, status = await input_queue.get()
            yield indata, status


async def write_chunks(stream):
    # This connects the raw audio chunks generator coming from the microphone
    # and passes them along to the transcription stream.
    async for chunk, status in mic_stream():
        await stream.input_stream.send_audio_event(audio_chunk=chunk)
    await stream.input_stream.end_stream()


async def basic_transcribe():
    await start_transcription()
async def start_transcription():
    try:
        client = TranscribeStreamingClient(region="us-west-2")
        stream = await client.start_stream_transcription(
            language_code="en-US",
            media_sample_rate_hz=16000,
            media_encoding="pcm",
        )
        handler = MyEventHandler(stream.output_stream)
        await asyncio.gather(write_chunks(stream), handler.handle_events())
    except amazon_transcribe.exceptions.BadRequestException:
        print("Transcription stream timeout. Restarting transcription...")
        await start_transcription()  # Restart transcription on timeout
    except Exception as e:
        print(f"An error occurred: {e}")
        
def play_video_moviepy(file_path):
    video = VideoFileClip(file_path)
    video.preview()

loop = asyncio.get_event_loop()

if __name__ == "__main__":
    asyncio.run(basic_transcribe())