import librosa
import numpy as np
import openai
import whisper
from googletrans import Translator
from moviepy.editor import AudioFileClip
from text_model import predict_text_mod


def duration_check(audio_file):
    audio = AudioFileClip(audio_file)
    audio_duration = audio.duration
    if audio_duration > 45:
        return False
    else:
        return True


def transcribe_audio(audio_file):
    print("Transcribing the Audio")
    audio, sr = librosa.load(audio_file)
    audio /= np.max(np.abs(audio))
    model = whisper.load_model("base")
    result = model.transcribe(audio)
    return result


def translate_text(text, target_language):
    print("Translating the Text")
    translator = Translator()
    translation = translator.translate(text, dest=target_language)
    return translation.text


def audio_moderate(audio_file):
    length = duration_check(audio_file)
    if length == False:
        return "Please upload audio file having length of duration less than 45 seconds"
    else:
        transcribed_result = transcribe_audio(audio_file)
        print(transcribed_result)
        if not transcribed_result["text"]:
            return "No text found in the audio"
        translated_text = translate_text(transcribed_result["text"], "en")
        MODERATION_CLASS = predict_text_mod(translated_text)
        return MODERATION_CLASS


if __name__ == "__main__":
    audio_file = input("Enter Audio File: ")

    result = audio_moderate(audio_file)
    print(result)
