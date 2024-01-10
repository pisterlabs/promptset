# -*- coding: iso-8859-9 -*-
# -*- coding: utf-8 -*-
import logging
import contextlib
import logging
import os
import sys
import time
import noisereduce as nr
from scipy.io import wavfile

import openai
import subprocess
import traceback
import urllib.request
import urllib.request
import wave
from pathlib import Path

import requests
import speech_recognition as sr
from pyannote.audio import Pipeline
from pydub import AudioSegment
from speechbrain.pretrained import SpeakerRecognition

logger = logging.getLogger(__name__)

# Global Variables
pyannote_url = "https://huggingface.co/pyannote/"
speechbrain_url = "https://huggingface.co/speechbrain/"
pyannote_dir = "pyannote_models"
speechbrain_dir = "speechbrain_models"


class SpeechDiarization:

    def __init__(self):
        self.pipeline = None
        self.verification = None

    def load_diarization_model(self):
        self._load_model_pyannote("pyannote/speaker-diarization@2.1", pyannote_dir)

    def load_verify_model(self):
        self._load_model_speechbrain("speechbrain/spkrec-ecapa-voxceleb", speechbrain_dir)

    def _load_model_pyannote(self, model_name, cache_dir):
        try:
            if not os.path.exists(cache_dir):
                os.mkdir(cache_dir)
            if not os.path.exists(os.path.join(cache_dir, "speaker-diarization@2.1")):
                with contextlib.closing(urllib.request.urlopen(pyannote_url)) as url:
                    self.pipeline = Pipeline.from_pretrained(model_name,
                                                             use_auth_token="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",
                                                             cache_dir=cache_dir)
            else:
                self.pipeline = Pipeline.from_pretrained(os.path.join(cache_dir, "speaker-diarization@2.1"))
        except Exception as e:
            print(f"Error: Unable to initialize models. {str(e)}")

    def _load_model_speechbrain(self, model_name, cache_dir):
        if not os.path.exists(cache_dir):
            os.mkdir(cache_dir)
        if not os.path.exists(os.path.join(cache_dir, "spkrec-ecapa-voxceleb")):
            with contextlib.closing(urllib.request.urlopen(speechbrain_url)) as url:
                self.verification = SpeakerRecognition.from_hparams(source=model_name,
                                                                    savedir=os.path.join(cache_dir,
                                                                                         model_name.split("/")[-1]))
        else:
            self.verification = SpeakerRecognition.from_hparams(source=model_name,
                                                                savedir=os.path.join(cache_dir,
                                                                                     model_name.split("/")[-1]))



class SpeechRecognizer:
    def __init__(self, speech_diarization, name):
        self.profileSpeech = f"C:/xampp/htdocs/Bitirme/Gorisim/Users/{name}/profile/"
        self.r = sr.Recognizer()
        self.pipeline = speech_diarization.pipeline
        self.verification = speech_diarization.verification

    def recognize(self, filename):
        input_file = Path(filename)
        output_file = Path("C:/xampp/htdocs/Bitirme/outputfile.wav")
        combined_file = Path("C:/xampp/htdocs/Bitirme/combined.wav")

        with wave.open(str(input_file), "rb") as infile:
            nchannels = infile.getnchannels()
            sampwidth = infile.getsampwidth()
            framerate = infile.getframerate()
            try:
                combine = AudioSegment.silent(duration=0)

                # Recognize speech
                output = self.pipeline(input_file, min_speakers=1, max_speakers=6)
                for turn, _, speaker in output.itertracks(yield_label=True):
                    if os.path.isdir(self.profileSpeech):
                        start = turn.start
                        end = turn.end
                        infile.setpos(int(start * framerate))
                        data = infile.readframes(int((end - start) * framerate))
                        with wave.open(str(output_file), "wb") as outfile:
                            outfile.setnchannels(nchannels)
                            outfile.setsampwidth(sampwidth)
                            outfile.setframerate(framerate)
                            outfile.setnframes(int(len(data) / sampwidth))
                            outfile.writeframes(data)
                        # verify the temperorary file
                        counter = 0

                        try:
                            with os.scandir(self.profileSpeech) as entries:
                                for entry in entries:
                                    if counter >= 2:
                                        break
                                    score, prediction = self.verification.verify_files(str(output_file),
                                                                                       f'C:/xampp/htdocs/Bitirme/Gorisim/Users/{name}/profile/{entry.name}')
                                    if prediction:
                                        counter += 1
                        except RuntimeError as k:
                            continue
                        if counter >= 2:
                            the_result_audio_file = AudioSegment.from_wav(str(output_file))
                            combine += the_result_audio_file
                        else:

                            continue
                    else:
                        if speaker[9] == "0":
                            start = turn.start
                            end = turn.end
                            infile.setpos(int(start * framerate))
                            data = infile.readframes(int((end - start) * framerate))
                            with wave.open(str(output_file), "wb") as outfile:
                                outfile.setnchannels(nchannels)
                                outfile.setsampwidth(sampwidth)
                                outfile.setframerate(framerate)
                                outfile.setnframes(int(len(data) / sampwidth))
                                outfile.writeframes(data)
                            # verify the temperorary file
                            the_result_audio_file = AudioSegment.from_wav(str(output_file))
                            combine += the_result_audio_file
                        else:
                            continue

                combine.export(str(combined_file), format="wav")
                with sr.AudioFile(str(combined_file)) as the_combined_data:
                    audio = self.r.record(the_combined_data)

                text = self.r.recognize_google(audio, language="tr-tr")
                return text
            except sr.UnknownValueError as e:
                with sr.AudioFile(str(input_file)) as input_file:
                    audio = self.r.record(input_file)
                text = self.r.recognize_google(audio, language="tr-tr")
                return text
            except sr.RequestError as e:
                with sr.AudioFile(str(input_file)) as input_file:
                    audio = self.r.record(input_file)
                text = self.r.recognize_google(audio, language="tr-tr")
                return text
            except ValueError as e:
                return e


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    file_path = sys.argv[1]

    name = sys.argv[2]
    # create an instance of SpeechDiarization
    speech_diarization = SpeechDiarization()
    speech_diarization.load_diarization_model()
    speech_diarization.load_verify_model()
    recognizer = SpeechRecognizer(speech_diarization, name=name)

    text = recognizer.recognize(filename=file_path)

    openai.api_key = "yout key"
    prompt = f'"{text}" cümlesindeki kelimelerin köklerini aralarýna boþluk býrakarak yazdýr. Sadece kelime köklerini yazdýr.Sadece kelime köklerini yazdýr. Sadece kelime köklerini yazdýr.Sadece kelime köklerini yazdýr.Sadece kelime köklerini yazdýr.'

    try:
        result = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        sentence = (result['choices'][0]['message']['content'])
        sentence = sentence.replace("-", "")
        sentence = sentence.replace(",", "")
        sentence = sentence.replace("\n", "")
        print(sentence)
    except:
        print(text)
    os.remove(file_path)
