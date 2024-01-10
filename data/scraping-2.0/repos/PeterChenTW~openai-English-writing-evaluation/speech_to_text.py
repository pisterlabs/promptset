import speech_recognition as sr
import openai 

openai.api_key = ''
recognizer = sr.Recognizer()

def free_model_audio_to_text(recording_file_path):
    with sr.AudioFile(recording_file_path) as source:
        audio_data = recognizer.record(source)

    # Recognize the audio
    text = recognizer.recognize_google(audio_data)
    return text


def openai_model_audio_to_text(recording_file_path):
    with open(recording_file_path, 'rb') as audio_file:
        transcript = openai.Audio.transcribe(
            model='whisper-1', 
            file=audio_file,
            temperature=0.3,
            prompt="Umm, let me think like, hmm... Okay, here's what I'm, like, thinking. I love play ball.",
            language='en'
        )

    return transcript['text']
    