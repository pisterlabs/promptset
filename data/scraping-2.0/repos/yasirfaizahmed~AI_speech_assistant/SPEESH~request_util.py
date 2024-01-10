"""
File: speesh.py
Author: yaseer faiz ahmed
Date: May 6, 2023
Description: SPEECH is an assistant powered by google-cloud, OpenAI APIs.
"""


import os
from google.cloud import speech_v1p1beta1 as speech
import openai as ai
from google.cloud import texttospeech

from audio_util import AudioUtil

# Audio recording parameters
CHANNELS = 2
RATE = 44100


class Request():
  def __init__(self):
    # Create a Client instance
    self.speech_client = speech.SpeechClient()
    self.text_client = texttospeech.TextToSpeechClient()

  def speech_to_text(self, audio_file: str):

    if os.path.exists(audio_file) is False:
      return False
    # Audio file
    audio_file = open(audio_file, 'rb')

    audio = speech.RecognitionAudio(content=audio_file.read())
    # Configure the audio settings
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code="en-IN",
        audio_channel_count=CHANNELS
    )

    # Call the Speech-to-Text API
    response = self.speech_client.recognize(config=config, audio=audio)

    # Print the transcript
    transcript = ""
    confidence = None
    for result in response.results:
      transcript = result.alternatives[0].transcript
      confidence = result.alternatives[0].confidence

    if confidence is None:
      confidence = 0
    return transcript, confidence

  def ask_openai(self, user_prompt: str,
                 engine='text-davinci-003',
                 temperature=0.5,
                 max_tokens=100) -> str:
    # secret key
    secret = os.environ['API_KEY']
    ai.api_key = secret

    completions = ai.Completion.create(engine=engine,            # Determines the quality, speed, and cost.
                                       temperature=temperature,  # Level of creativity in the response
                                       prompt=user_prompt,       # What the user typed in
                                       max_tokens=max_tokens,    # Maximum tokens in the prompt AND response
                                       n=1,                      # The number of completions to generate
                                       stop=None)
    return completions.choices[0].text.strip()

  def text_to_speech(self, text: str):
    # Set the text input to be synthesized
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Build the voice request, select the language code ("en-US") and the ssml
    # voice gender ("neutral")
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-IN", ssml_gender=texttospeech.SsmlVoiceGender.FEMALE
    )

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Perform the text-to-speech request on the text input with the selected
    # voice parameters and audio file type
    response = self.text_client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    audio_file = AudioUtil()._calibrate_audio_file_path(mode='out', format='mp3')

    # The response's audio_content is binary.
    with open(audio_file, "wb") as out:
      # Write the response to the output file.
      out.write(response.audio_content)

    return audio_file


if __name__ == '__main__':
  # speech_to_text(audio_file='/home/xd/Documents/python_codes/AI_speesh_assistant/_AUDIOs/May-07-2023_15-18-51.flac')
  # print(ask_openai(user_prompt="hello there.."))
  # text_to_speech(text="hello there")
  pass
