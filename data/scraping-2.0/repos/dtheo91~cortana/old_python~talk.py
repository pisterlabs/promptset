# Example: reuse your existing OpenAI setup
import openai
from elevenlabs import Voice, VoiceSettings, generate, play
from elevenlabs import set_api_key

import pyaudio
from vosk import Model, KaldiRecognizer
import keyboard
import numpy as np
from queue import Queue
from threading import Thread
import json

import sys

def jarvis():
    set_api_key("5fdf2db6d11c7a73be4ad872bc784ebe")
    client = openai.OpenAI(base_url="http://localhost:1234/v1", api_key="not-needed")

    model = Model(model_name="vosk-model-en-us-0.22")
    messages = Queue()
    recordings = Queue()

    CHANNELS = 1
    FRAME_RATE = 16000
    RECORD_SECONDS = 5
    AUDIO_FORMAT = pyaudio.paInt16

    rec = KaldiRecognizer(model, FRAME_RATE)
    rec.SetWords(True)

    def chat_with_gpt(prompt):
        """Chat with the LM-Studio engine."""
        response = client.chat.completions.create(
            model = "deepcoded/DeepCoder",
            messages = [{"role":"user", "content":prompt}],
        )
        return response.choices[0].message.content

    def record_microphone(chunk=1024):
        p = pyaudio.PyAudio()

        stream = p.open(format=AUDIO_FORMAT,
                        channels=CHANNELS,
                        rate=FRAME_RATE,
                        input=True,
                        input_device_index=1,
                        frames_per_buffer=chunk)

        frames = []

        while not messages.empty():
            data = stream.read(chunk)
            frames.append(data)
            if len(frames) >= (FRAME_RATE * RECORD_SECONDS) / chunk:
                recordings.put(frames.copy())
                frames = []

        stream.stop_stream()
        stream.close()
        p.terminate()

    def speech_recognition():
        frames = recordings.get()
        rec.AcceptWaveform(b''.join(frames))
        result = rec.Result()
        text = json.loads(result)["text"]
        return text

    def start_recording():
        messages.put(True)
        record = Thread(target=record_microphone)
        record.start()

    def stop_recording():
        prompt = speech_recognition()
        messages.get()
        response = chat_with_gpt(prompt)

        sys.stdout.write(json.dumps({"message": prompt, "response": response}))
        sys.stdout.flush()

        audio = generate(
        text=response,
            voice=Voice(
                voice_id = "P0edYnhQCCbHWG5n6UeC",
                settings=VoiceSettings(stability=0.5, similarity_boost=0.35, style=0.0, use_speaker_boost=True)
            ),
            model="eleven_turbo_v2"
        )

        play(audio)

    def on_key_event(e):
        if e.name == "alt" and e.event_type == keyboard.KEY_DOWN:
            start_recording()
            
        elif e.name == "alt" and e.event_type == keyboard.KEY_UP:
            stop_recording()

    # Register the callback for F1 key events
    keyboard.hook(on_key_event)

    # Keep the script running until manually interrupted
    keyboard.wait("esc")

if __name__ == "__main__":
    jarvis()


    

