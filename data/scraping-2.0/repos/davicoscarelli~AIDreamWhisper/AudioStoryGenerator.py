# AudioStoryGenerator.py

import openai
from pyht import Client
from dotenv import load_dotenv
from pyht.client import TTSOptions
import wave
import numpy as np
import soundfile as sf
from scipy.signal import resample
import io
import os


load_dotenv()


openai.api_key =  os.getenv("OPENAI_API_KEY")


client = Client(os.getenv("PLAYHT_USER_ID"), os.getenv("PLAYHT_API_KEY"))


options = TTSOptions(voice="s3://voice-cloning-zero-shot/d9ff78ba-d016-47f6-b0ef-dd630f59414e/female-cs/manifest.json")

def get_story_from_openai(topic: str) -> str:
    prompt = (f"Given the scientific foundation of enhancing learning through auditory stimuli during REM sleep, "
              f"craft a calming, educational story designed for nighttime learning that reinforces/teaches the key concepts of '{topic}'. "
              f"The story should be structured to enhance memory retention and deepen understanding of the topic.")
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=500
    )
    return response.choices[0].text.strip()




def split_into_chunks(text, max_chunk_length):
    
    sentences = text.split('.')
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        
        if len(current_chunk) + len(sentence) <= max_chunk_length:
            current_chunk += sentence + '.'
        else:
            
            chunks.append(current_chunk)
            current_chunk = sentence + '.'
    
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def generate_audio_from_text(text):
    audio_data = b""
    max_chunk_length = 250
    chunks = split_into_chunks(text, max_chunk_length)

    sample_rate = None 
    for chunk in chunks:
        print(chunk)
        chunk_audio_data = b""
        for audio_chunk in client.tts(chunk, options):
            chunk_audio_data += audio_chunk
        
        if sample_rate is None:
            with wave.open(io.BytesIO(chunk_audio_data), 'rb') as w:
                sample_rate = w.getframerate()
        
        chunk_samples = np.frombuffer(chunk_audio_data, dtype=np.int16)
        
        chunk_samples = chunk_samples = fade(chunk_samples, 0.01, 0.01, 44100)
        
        audio_data += chunk_samples.tobytes()
    return audio_data, sample_rate


def fade(samples, fade_in_duration, fade_out_duration, sample_rate, fade_start=0, fade_end=1):
    fade_in_samples = int(fade_in_duration * sample_rate)
    fade_out_samples = int(fade_out_duration * sample_rate)
    
    samples = np.copy(samples)
    
    samples[:fade_in_samples] = np.multiply(samples[:fade_in_samples], np.linspace(fade_start, fade_end, fade_in_samples))
    
    samples[-fade_out_samples:] = np.multiply(samples[-fade_out_samples:], np.linspace(fade_end, fade_start, fade_out_samples))
    return samples


def mix_wav_files(file1, file2, output_file):
    
    with wave.open(file1, 'rb') as w1, wave.open(file2, 'rb') as w2:
        
        assert w1.getframerate() == w2.getframerate()
        assert w1.getnchannels() == w2.getnchannels()

        frames1 = w1.readframes(w1.getnframes())
        frames2 = w2.readframes(w2.getnframes())

        
        samples1 = np.frombuffer(frames1, dtype=np.int16)
        samples2 = np.frombuffer(frames2, dtype=np.int16)

        
        samples1 = fade(samples1, 0.5, 0.5, 44100)

        
        while len(samples2) < len(samples1):
            samples2 = np.concatenate((samples2, samples2))

        
        samples2 = samples2[:len(samples1)]
        samples2 = fade(samples2, 0, 1, 44100)

        
        mixed_samples = (samples1 * 0.5 + samples2 * 0.5).astype(np.int16)

        
        with wave.open(output_file, 'wb') as out:
            out.setnchannels(w1.getnchannels())
            out.setsampwidth(w1.getsampwidth())
            out.setframerate(w1.getframerate())
            out.writeframes(mixed_samples.tobytes())


def resample_wav_file(input_file, output_file, target_sr):
    with wave.open(input_file, 'rb') as w:
        orig_sr = w.getframerate()
        audio = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    
    resampled_audio = resample(audio, int(len(audio) * target_sr / orig_sr))
    resampled_audio = np.int16(resampled_audio)

    with wave.open(output_file, 'wb') as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(target_sr)
        w.writeframes(resampled_audio.tobytes())

class AudioStoryGenerator:
    def __init__(self):
        pass

    def generate(self, topic):
        calming_story = get_story_from_openai(topic)
        audio_data, sample_rate = generate_audio_from_text(calming_story)
        audio_samples = np.frombuffer(audio_data, dtype=np.int16)
        sf.write("output.wav", audio_samples, sample_rate)
        resample_wav_file("output.wav", "output_resampled.wav", 44100)
        resample_wav_file("background.wav", "background_resampled.wav", 44100)
        mix_wav_files("output_resampled.wav", "background_resampled.wav", "static/output_with_music.wav")
        return "output_with_music.wav"
