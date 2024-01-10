import speech_recognition as sr
import librosa
import torch
import whisper
from transformers import Wav2Vec2ForCTC, AutoProcessor
from time import process_time
from datasets import load_dataset
from evaluate import load
import openai
from dotenv import find_dotenv, load_dotenv
import os

def openai_s2t(file):
  audio_file = open(file, 'rb')
  # audio_file = open('./MLKDream.flac', 'rb')
  transcript = openai.Audio.transcribe('whisper-1', audio_file, response_format='text')
  return transcript

audio_files = ["MLKDream.flac"]

def SpeechRecognition_s2t(audio_files):
  transcriptions = []

  for audio_file in audio_files:
    r = sr.Recognizer()
    with sr.AudioFile(audio_file) as source:
      audio = r.record(source)
    transcription = r.recognize_google(audio)
    transcriptions.append(transcription)

  return transcriptions

def Wav2Vec2_s2t(audio_files, rate=16000):
  transcriptions = []
  processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
  model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

  for audio_file in audio_files:
    audio, rate = librosa.load(audio_file, sr=rate)
    inputs = processor([audio], sampling_rate=rate, return_tensors="pt")

    with torch.no_grad():
      logits = model(**inputs).logits
      predicted_ids = torch.argmax(logits, dim=-1)

    transcription = processor.batch_decode(predicted_ids)[0]
    transcriptions.append(transcription)

  return transcriptions

def whisper_s2t(audio_files):
  transcriptions = []
  model = whisper.load_model("base")

  for audio_file in audio_files:
    result = model.transcribe(audio_file)
    transcriptions.append(result["text"])

  return transcriptions

# transcriptions = SpeechRecognition_s2t(audio_files)
# display_transcript(transcriptions)

# transcriptions = Wav2Vec2_s2t(audio_files)
# display_transcript(transcriptions)

# transcriptions = whisper_s2t(audio_files)
# display_transcript(transcriptions)

# def speech2text(audio_file):
#   model = whisper.load_model("base")
#   result = model.transcribe(audio_file)
#   return result["text"]

# transcript = speech2text(audio_file)

# def check_asr_accuracy(audio_files, references, model_func):
#   predictions = []
#   wer_scores = []
#   for audio_path, reference in zip(audio_files, references):
#     transcript = model_func(audio_path)
#     predictions.append(transcript)
#     wer = load("wer")
#     wer_score = wer.compute(predictions=[transcript], references=[reference])
#     wer_scores.append(wer_score)
#   final_score = wer.compute(predictions=predictions, references=references)
#   return (predictions, wer_scores, final_score)


# def whisper_check(audio_file):
#   model = whisper.load_model("base")
#   result = model.transcribe(audio_file)["text"]
#   transcript = result.replace(".", "").replace(",", "").replace("?", "").replace("!", "").upper()[1:]
#   return transcript


# def Wav2Vec2_check(audio_file, rate=16000):
#   processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
#   model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")

#   audio, rate = librosa.load(audio_file, sr=rate)
#   inputs = processor([audio], sampling_rate=rate, return_tensors="pt")

#   with torch.no_grad():
#     logits = model(**inputs).logits
#     predicted_ids = torch.argmax(logits, dim=-1)

#   transcription = processor.batch_decode(predicted_ids)[0]

#   return transcription


# ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")

def setup():
  load_key()
  transcription = openai_s2t()
  print(transcription)

if __name__ == '__main__':
  setup()
