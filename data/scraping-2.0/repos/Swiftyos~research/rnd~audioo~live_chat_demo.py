import sounddevice as sd
import numpy as np
import whisper
import openai
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
from scipy.io.wavfile import write
import threading
import time

# Initialize models and processors
whisper_model = whisper.load_model("base")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0)

interruption_detected = False
progress_marker = 0  # Track progress of playback in terms of samples


def list_devices():
    """List all available audio devices."""
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(
            f"{i}: {device['name']} (Channels: {device['max_input_channels']}/{device['max_output_channels']})"
        )


def monitor_user_interruption(samplerate=16000, threshold=200, device_index=0):
    global interruption_detected
    with sd.InputStream(
        samplerate=samplerate, channels=1, device=device_index
    ) as stream:
        while not interruption_detected:
            audio_chunk, _ = stream.read(samplerate)
            if np.abs(audio_chunk).mean() > threshold:  # if user interrupts
                interruption_detected = True


def play_and_monitor(speech, samplerate=16000, device_index=1):
    global interruption_detected, progress_marker
    threading.Thread(
        target=monitor_user_interruption
    ).start()  # Start monitoring in a separate thread

    # Ensure the speech waveform is mono for playback
    # if len(speech.shape) > 1 and speech.shape[1] > 1:
    #     speech = np.mean(speech, axis=1)  # Convert to mono if stereo

    for i in range(progress_marker, len(speech), samplerate):
        if interruption_detected:
            break
        sd.play(speech[i : i + samplerate], samplerate=samplerate, device=device_index)
        sd.wait()
        progress_marker += samplerate

    if interruption_detected:
        sd.stop()  # If user interrupts, stop playing


def listen_and_respond(messages, selected_microphone_index, selected_speaker_index):
    global interruption_detected  # <-- Add this line

    # Settings for listening
    chunk_duration = 1
    silence_threshold = 50
    silence_duration = 1.0
    consecutive_silent_chunks = int(silence_duration / chunk_duration)
    samplerate = 44100
    silent_count = 0
    recordings = []

    with sd.InputStream(
        samplerate=samplerate, channels=1, device=selected_microphone_index
    ) as stream:
        while True:
            audio_chunk, overflowed = stream.read(int(samplerate * chunk_duration))
            recordings.append(audio_chunk)
            if np.abs(audio_chunk).mean() < silence_threshold:
                silent_count += 1
            else:
                silent_count = 0
            if silent_count >= consecutive_silent_chunks:
                break

    recording = np.concatenate(recordings, axis=0)
    wav_file = "temp_audio.wav"
    write(wav_file, samplerate, recording)
    result = whisper_model.transcribe(wav_file)
    transcribed_text = result["text"]
    print(f"User: {transcribed_text}")

    if transcribed_text == "":
        return
    messages.append({"role": "user", "content": transcribed_text})

    openai.api_key = "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"
    msgs = [
        {"role": "system", "content": "You are a helpful ai assistant"},
        {"role": "user", "content": "transcribed_text"},
        {"role": "assistant", "content": "transcribed_text"},
    ]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo", messages=msgs, temperature=0.5, max_tokens=256
    )
    ai_response = response.choices[0].message.content
    print(f"Agent: {ai_response}")

    inputs = processor(text=ai_response, return_tensors="pt")
    spectrogram = model.generate_speech(inputs["input_ids"], speaker_embeddings)
    speech = model.generate_speech(
        inputs["input_ids"], speaker_embeddings, vocoder=vocoder
    )
    messages.append({"role": "system", "content": ai_response})

    # Play AI's response and monitor for interruption
    play_and_monitor(speech, device_index=selected_speaker_index)

    if interruption_detected:
        said_so_far = speech[:progress_marker]
        # Optionally save or transcribe `said_so_far` here
        interruption_detected = False
        listen_and_respond(messages, selected_microphone_index, selected_speaker_index)

    return messages


if __name__ == "__main__":
    # List available devices and prompt user for selection
    list_devices()
    selected_microphone_index = int(
        input("Enter the index of the microphone you want to use: ")
    )
    selected_speaker_index = int(
        input("Enter the index of the speaker you want to use: ")
    )

    # Main loop
    while True:
        messages = []
        messages = listen_and_respond(
            messages, selected_microphone_index, selected_speaker_index
        )
