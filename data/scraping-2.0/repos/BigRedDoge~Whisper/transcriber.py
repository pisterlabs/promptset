import openai
import pyaudio
import wave


def transcribe(frames, path):
    save_audio(frames, path)
    transcript = openai.Audio.transcribe("whisper-1", open(path, "rb"))
    return transcript["text"]

def save_audio(frames, path):
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(44100)
        wf.writeframes(b''.join(frames))