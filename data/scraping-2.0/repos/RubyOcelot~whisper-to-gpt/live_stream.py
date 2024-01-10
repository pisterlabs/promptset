import os
import pyaudio
import openai
import io
import threading
from pydub import AudioSegment
import time
from gpt_translate import translate_by_gpt3_5_turbo
from language_context_config import *

openai.api_key = os.getenv("OPENAI_API_KEY")

# Set audio data parameters
sample_rate = 48000  # Hz
sample_width = 2  # Bytes (16-bit integer)
num_channels = 1  # Mono audio
chunk_size = 1024  # Bytes
record_segment_time = 10  # Seconds
# transcript_step_time = 2  # Seconds


context_config = get_lan_config()


def whisper_transcribe_segment(file_like_segment, prompt=""):
    # Transcribe the chunk of audio data
    result = openai.Audio.transcribe_raw(
        "whisper-1",
        file_like_segment,
        "file.mp3",
        prompt=prompt,
    )
    return result["text"]

def get_transcript_from_segment(segment_data,segment_id):

    # Convert audio data to AudioSegment object
    byte_stream = io.BytesIO(segment_data)
    audio_segment = AudioSegment.from_file(byte_stream, format='raw', frame_rate=sample_rate, sample_width=sample_width, channels=num_channels)

    # Export audio segment to desired format
    file_format = 'mp3'  # Example format
    output_stream = io.BytesIO()
    # output_file_base_name = "tmp/live_stream_3"
    # audio_segment.export(output_file_base_name+"_"+str(segment_id)+".mp3", format=file_format)
    audio_segment.export(output_stream, format=file_format)

    # print("Segment ", segment_id, ": Transcribing audio...")
    t1 = time.time()
    text = whisper_transcribe_segment(output_stream, prompt=context_config.transcribe_prompt)
    t2 = time.time()
    print("Segment ", segment_id, ": Transcribed audio in {} seconds".format(t2-t1))
    print("Segment ", segment_id, ": text:\n", text)


    translated_text = translate_by_gpt3_5_turbo(text=text, language_from=context_config.language_from, language_keep=context_config.language_keep, language_contain=context_config.language_contain, other_translate_context=context_config.other_translate_context)
    t3 = time.time()
    print("Segment ", segment_id, ": Translated text in {} seconds".format(t3-t2))
    print("Segment ", segment_id, ": translated text:\n", translated_text)

def process_stream_in_seg():

    # Define the stream and start recording
    # print(pyaudio.PyAudio().get_device_info_by_index(0)['defaultSampleRate'])
    stream = pyaudio.PyAudio().open(format=pyaudio.paInt16, channels=num_channels, rate=sample_rate, input=True, frames_per_buffer=chunk_size)
    stream.start_stream()

    # Transcribe the audio stream
    transcript = ""
    count = 0
    while True:
        print(f"Capturing segment {count+1}")
        frames = []
        for j in range(0, int(sample_rate / chunk_size * record_segment_time)):
            data = stream.read(chunk_size)
            frames.append(data)
        segment_data=b''.join(frames)

        count += 1
        t = threading.Thread(target=get_transcript_from_segment, args=(segment_data, count))
        t.start()

    # Stop the stream and close it
    stream.stop_stream()
    stream.close()

    # Print the final transcript
    print(transcript)

if __name__ == "__main__":
    process_stream_in_seg()
