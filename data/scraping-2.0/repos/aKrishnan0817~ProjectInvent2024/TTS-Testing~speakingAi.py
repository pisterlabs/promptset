import speech_recognition as sr
from openai import OpenAI
import sys
import os
import wave

sys.path.append('/')
import sensitiveData
client = OpenAI(api_key=sensitiveData.apiKey)

'''This speach regonizer works by saving the your speech as an audio file, sending that to open ai's whisper api
then deleting the file and repeating. This seems to be faster than using the whisper model'''


# Initialize the SpeechRecognition recognizer
recognizer = sr.Recognizer()

# Create a function to transcribe speech using OpenAI API
def transcribe_speech(audio_file_path):
    audio_file= open(audio_file_path, "rb")
    transcript = client.audio.transcriptions.create(
      model="whisper-1",
      file=audio_file
    )
    print(transcript.text)

# Create a function for live speech recognition
def live_transcriber():
    with sr.Microphone() as source:
        print("Say something...")
        try:
            while True:
                audio_data = recognizer.listen(source)

                # Save the audio file temporarily
                audio_file_path = "temp_audio.wav"
                with open(audio_file_path, "wb") as temp_audio_file:
                    temp_audio_file.write(audio_data.get_wav_data())

                # Transcribe using OpenAI API
                transcribe_speech(audio_file_path)
        except sr.UnknownValueError:
            print("Speech Recognition could not understand audio.")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
        finally:
             #Clean up: remove temporary audio file
            if os.path.exists(audio_file_path):
                os.remove(audio_file_path)

# Run the live transcriber
live_transcriber()
