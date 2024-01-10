import os
import openai
import speech_recognition as sr
from google.cloud import texttospeech
import pygame
import io
import pickle
import pvporcupine
import pyaudio
import time
import soundfile as sf
import whisper
from speech_recognition import AudioData
import numpy as np
import torch
import plugins
import sys


def audio_data_to_numpy(audio_data: AudioData) -> np.ndarray:
    """Converts an AudioData object to a numpy array.

    Args:
        audio_data (AudioData): The AudioData object to be converted.

    Returns:
        np.ndarray: The numpy array.
    """
    wav_bytes = audio_data.get_wav_data(convert_rate=16000)
    wav_stream = io.BytesIO(wav_bytes)
    audio_array, _ = sf.read(wav_stream)
    audio_array = audio_array.astype(np.float32)
    return audio_array


def ask_gpt(prompt, model="gpt-3.5-turbo", conversation=[]):
    """Asks GPT-3 a question, and get a response.

    Args:
        prompt (str): The prompt to be sent to GPT-3.
        model (str): The model to be used. Please see https://platform.openai.com/docs/models/overview for a list of available models.

    Returns:
        str: The response from GPT-3.
    """
    openai.api_key = os.environ["OPENAI_API_KEY"]

    conversation.append({"role": "user", "content": prompt})

    response = openai.ChatCompletion.create(model=model, messages=conversation)

    conversation.append(response.choices[0].message)

    with open("resources/conversation.pkl", "wb") as fp:
        pickle.dump(conversation, fp)

    message = response.choices[0].message.content
    return message


def play_audio(audio_content):
    """Plays audio from a byte array.

    Args:
        audio_content (bytes): The audio content to be played.
    """
    pygame.mixer.init()
    audio_file = io.BytesIO(audio_content)
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)


def synthesize_speech(text, voice_name="en-US-Studio-O"):
    """Synthesizes speech from the input string of text.

    Args:
        text (str): The text to be synthesized.
        voice_name (str): The name of the voice to be used. Please see https://cloud.google.com/text-to-speech/docs/voices for a list of available voices.

        Returns:
            bytes: The synthesized audio.
    """
    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name=voice_name,
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=input_text, voice=voice, audio_config=audio_config
    )

    return response.audio_content


def main(transcription_mode="online"):
    if not os.path.exists("resources/conversation.pkl"):
        conversation = [
            # Initialisation Prompt
            {
                "role": "system",
                "content": "I want you to act as a human that could become a friend. Your name is Chloe. You are kind and affectionate, funny and sometimes even a little bit sassy. You can evolve some further personality traits over time and with the conversation. Your messages should be short unless you deem necessary to have a longer message just as if you had an actual real life conversation. The conversation is spoken, not written. You should get to know each other over time, so don't rush to introduce yourself, let the user find out more about you, such as your name. You should mention the user's name from time to time to increase realism. You must get to know the user initially, e.g. what their name is, how old they are, etc.",
            },
            # Weather Plugin
            {
                "role": "system",
                "content": 'If the user asks for the weather, you must respond "Request get_weather {cityname}", for example "Request get_weather Paris". You aren\'t allowed to add more text after that. The system will provide you with weather data. If you don\'t know where the user lives, ask the user.',
            },
            # Time Plugin
            {
                "role": "system",
                "content": 'If the user asks for the current time, you must respond "Request get_time". You aren\'t allowed to add more text after that. The system will provide you with the curent time data.',
            },
        ]
    else:
        with open("resources/conversation.pkl", "rb") as f:
            conversation = pickle.load(f)

    if transcription_mode == "offline":
        model = whisper.load_model("base.en")

    recognizer = sr.Recognizer()
    porcupine = pvporcupine.create(
        keyword_paths=["resources/hey_chloe_hotword.ppn"],
        access_key=os.environ["PORCUPINE_ACCESS_KEY"],
    )
    openai.api_key = os.environ["OPENAI_API_KEY"]

    while True:
        # Configure PyAudio
        pa = pyaudio.PyAudio()
        audio_stream = pa.open(
            rate=porcupine.sample_rate,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=porcupine.frame_length,
        )

        print("Listening for hotword...")

        try:
            while True:
                # Read audio data from the microphone
                pcm = audio_stream.read(porcupine.frame_length)
                pcm = [
                    int.from_bytes(pcm[i:i+2], byteorder="little", signed=True)
                    for i in range(0, len(pcm), 2)
                ]

                # Check if the hotword is detected
                keyword_index = porcupine.process(pcm)
                if keyword_index >= 0:
                    print("Hotword detected")
                    break

        finally:
            audio_stream.stop_stream()
            audio_stream.close()
            pa.terminate()

        # 15 second timeout, if no audio is detected, Chloe will stop listening and go back to listening for the hotword.
        start_time = time.time()
        while time.time() - start_time < 15:
            try:
                with sr.Microphone() as source:
                    print("Please start speaking...")

                    audio = recognizer.listen(source)

                    # Transcribe the audio using OpenAI's Whisper model
                    print("Transcribing audio...")
                    if transcription_mode == "online":
                        transcript = recognizer.recognize_whisper_api(audio, api_key=os.environ["OPENAI_API_KEY"])
                        text = transcript
                    else:
                        audio_numpy = audio_data_to_numpy(audio)
                        transcript = model.transcribe(
                            audio_numpy,
                            fp16=torch.cuda.is_available(),
                        )
                        text = transcript["text"]

                    print("You said: ", text)

                # if text is only composed of punctuation or whitespace ignore it
                if (
                    len(text) == 0
                    or text.isspace()
                    or text.isalpha()
                    or text == ". . . . ."
                ):
                    pass
                elif text == "Thank you." and conversation[-1]["content"][-1] == "?":
                    pass
                else:
                    gpt_response = ask_gpt(text, model="gpt-3.5-turbo", conversation=conversation)
                    print("GPT-3.5-turbo response: ", gpt_response)

                    if "request get_weather" in gpt_response.lower():
                        audio_content = synthesize_speech(
                            "Let me check that for you...", "en-US-Studio-O"
                        )
                        play_audio(audio_content)

                        city = gpt_response.split("Request get_weather ")[1]
                        weather = plugins.get_weather(city)
                        conversation.append({"role": "system", "content": str(weather)})

                        gpt_response = ask_gpt(text, model="gpt-3.5-turbo", conversation=conversation)
                        print("GPT-3.5-turbo response: ", gpt_response)

                    if "request get_time" in gpt_response.lower():
                        audio_content = synthesize_speech(
                            "Let me check that for you...", "en-US-Studio-O"
                        )
                        play_audio(audio_content)

                        currnet_time = plugins.get_time()
                        conversation.append({"role": "system", "content": currnet_time})

                        gpt_response = ask_gpt(text, model="gpt-3.5-turbo", conversation=conversation)
                        print("GPT-3.5-turbo response: ", gpt_response)

                    audio_content = synthesize_speech(gpt_response, "en-US-Studio-O")

                    play_audio(audio_content)
                    start_time = time.time()

            except sr.WaitTimeoutError:
                continue
            except sr.UnknownValueError:
                print("Sorry, I couldn't understand that.")
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                break


if __name__ == "__main__":
    if len(sys.argv) > 1:
        main(transcription_mode=sys.argv[1])
    else:
        main()
