from pydub import AudioSegment, silence
from moviepy.video.compositing.concatenate import concatenate_videoclips
import openai
import json
from scipy.io import wavfile
import numpy as np
import noisereduce as nr
import whisper
import os

def load_api_key():
    with open('creds/celtic-guru-247118-9e8cccc27c4a.json', 'r') as json_file:
        data = json.load(json_file)
        api_key = data.get('open_ai_key')
    return api_key


def remove_empty_space(temp_audio_path, export_path):
    audio = AudioSegment.from_wav(temp_audio_path)
    chunks = []
    silence_thresh = -50
    min_silence_duration = 1200
    audio_chunks = silence.split_on_silence(
        audio,
        min_silence_len=min_silence_duration,
        silence_thresh=silence_thresh
    )

    for chunk in audio_chunks:
        chunk_duration = len(chunk) / 1000  # Convert to seconds

        if chunk_duration > min_silence_duration / 1000:
            chunks.append(chunk)

    modified_audio = chunks[0]
    for chunk in chunks[1:]:
        silence_segment = AudioSegment.silent(duration=500)  # Half a second of silencze
        modified_audio += silence_segment + chunk

    cleaned_path = f"{export_path}/cleaned_audio.wav"
    modified_audio.export(cleaned_path, format="wav")
    # clean_noise(cleaned_path)
    # reduce_noise(cleaned_path)
    print("cleaned audio saved")

def clean_noise(path):
    rate, data = wavfile.read(path)
    original_shape = data.shape
    data = np.reshape(data, (2, -1))
    reduced_noise = nr.reduce_noise(
        y=data,
        sr=rate,
        stationary=True
    )
    wavfile.write(path, rate, reduced_noise.reshape(original_shape))



def reduce_noise(audio_path, noise_threshold=-40.0, fft_size=2048, overlap=0.5):
    # Load the audio file
    audio = AudioSegment.from_file(audio_path)

    # Convert overlap to integer value
    seek_step = int(overlap * audio.frame_rate)

    # Split the audio into chunks based on silence
    audio_chunks = silence.split_on_silence(audio, min_silence_len=500, silence_thresh=noise_threshold,
                                    keep_silence=200, seek_step=seek_step)

    # Initialize the output audio
    output_audio = AudioSegment.empty()

    # Concatenate the audio chunks with half a second of silence in between
    for i, chunk in enumerate(audio_chunks):
        if i > 0:
            output_audio += AudioSegment.silent(duration=500)  # Add half a second of silence
        output_audio += chunk

    # Export the output audio to a file
    output_audio.export(audio_path, format="wav")


def all_silent(temp_audio_path):
    audio = AudioSegment.from_wav(temp_audio_path)
    silence_thresh = -50
    min_silence_duration = 1200
    audio_chunks = silence.detect_silence(
        audio,
        min_silence_len=min_silence_duration,
        silence_thresh=silence_thresh
    )

    silent_chunks = []
    for start, end in audio_chunks:
        chunk_start_time = start / 1000  # Convert to seconds
        chunk_end_time = end / 1000  # Convert to seconds
        silent_chunks.append((chunk_start_time, chunk_end_time))

    return silent_chunks

def clips_to_video(video, chunks):
    clips = []

    for start_time, end_time in chunks:
        clip = video.subclip(start_time, end_time)
        clips.append(clip)

    return concatenate_videoclips(clips)

def get_viral_transcript(audio_path):
    openai.api_key = load_api_key()
    f = open(audio_path, "rb")
    transcript = openai.Audio.transcribe("whisper-1", f, prompt="Please transcribe only the first sentence")
    print(transcript)

def get_whisper_transcription(audio_path, folder_name, file_name=''):
    print("obtaining audio transcript")        
    if file_name != '':
        transcript_file = f"{folder_name}/whisper-{file_name}.json"
    else:
        transcript_file = f"{folder_name}/whisper-{folder_name}.json"

    if os.path.exists(transcript_file):
        with open(transcript_file, "r") as f:
            return json.load(f)
    else:
        desired_keys = ["id", "start", "end", "text"]
        condensed_data = []
        model = whisper.load_model("base")
        result = model.transcribe(audio_path)
        for segment in result["segments"]:
            condensed_segment = {key: segment[key] for key in desired_keys}
            condensed_data.append(condensed_segment)

        with open(transcript_file, 'w') as f:
            json.dump(condensed_data, f)
        return condensed_data

    # print(result)
