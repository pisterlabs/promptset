import requests
import numpy as np
from pydub import AudioSegment
import simpleaudio as sa
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QComboBox, QLabel, QTextEdit, QLineEdit, QMessageBox
from PyQt5.QtCore import Qt
from speechbrain.pretrained import Tacotron2, HIFIGAN
from elevenlabslib import *
import re
import sys

if sys.version_info < (3, 11):
    sys.exit("This script requires Python 3.11 or later")

def merge_audio_files(file_paths, output_path):
    combined = AudioSegment.empty()
    for file_path in file_paths:
        sound = AudioSegment.from_wav(file_path)
        combined += sound
    combined.export(output_path, format="mp3")

class WordDefinitionAndTTS:
    def __init__(self, api_key, base_url, tts='Tacotron2', dictionary='merriam', tts2_api_key=None, gpu_opt="No GPU", openai_api_key=None):
        self.api_key = api_key
        self.openai_api_key = openai_api_key
        self.base_url = base_url
        self.tts = tts
        self.dictionary = dictionary
        self.tts2_api_key = tts2_api_key
        self.gpu_opt = gpu_opt

    def get_definition(self, word):
        if self.dictionary == "gpt":
            import openai
            openai.api_key = self.openai_api_key
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert GRE vocab tutor. The user will input a word and you will return the word , it's definition, an easily rememberable way to memorize the word, an easily understandable example sentence"},
                    {"role": "user", "content": word}
                ]
            )
            content = completion.choices[0].message["content"]

            lines = content.split("\n")
            definition = None
            memorization_tip = None
            example_sentence = None

            for line in lines:
                if line.startswith("Definition:"):
                    definition = line.split("Definition:")[1].strip()
                elif line.startswith("Memorization tip:"):
                    memorization_tip = line.split("Memorization tip:")[1].strip()
                elif line.startswith("Example sentence:"):
                    example_sentence = line.split("Example sentence:")[1].strip()

            return definition + ". " + memorization_tip if definition and memorization_tip else None, example_sentence

        else:  # For 'merriam' dictionary
            response = requests.get(f'{self.base_url}{word}?key={self.api_key}')
            data = response.json()
            if not isinstance(data, list) or len(data) == 0 or not isinstance(data[0], dict):
                QMessageBox.warning(None, "Word not found or no definition available", f"Word not found or no definition available for: {word}")
                return None, None
            definitions = data[0].get('shortdef', [])
            # We need to ensure that the 'def', 'sseq', 'dt' fields exist in the dictionary to avoid KeyError
            if 'def' in data[0] and len(data[0]['def']) > 0 and 'sseq' in data[0]['def'][0] and len(data[0]['def'][0]['sseq']) > 0 and 'dt' in data[0]['def'][0]['sseq'][0][0][1] and len(data[0]['def'][0]['sseq'][0][0][1]['dt']) > 1 and len(data[0]['def'][0]['sseq'][0][0][1]['dt'][1][1]) > 0:
                example_sentence = data[0]['def'][0]['sseq'][0][0][1]['dt'][1][1][0]['t']
                example_sentence = example_sentence.replace('{it}', '').replace('{/it}', '')
                return definitions[0], example_sentence
            return None, None

    def text_to_speech(self, text, word):
        run_opts = {"device": "cuda"} if self.gpu_opt == "GPU" else {"device": "cpu"}
        if self.tts == "Tacotron2":
            #run_opts = {"device": "cuda"} if self.gpu_opt.currentText() == "GPU" else {"device": "cpu"}
            tacotron2 = Tacotron2.from_hparams(source="speechbrain/tts-tacotron2-ljspeech", savedir="tmpdir_tts", run_opts=run_opts)
            hifi_gan = HIFIGAN.from_hparams(source="speechbrain/tts-hifigan-ljspeech", savedir="tmpdir_vocoder", run_opts=run_opts)
            mel_output, mel_length, alignment = tacotron2.encode_text(text)
            waveforms = hifi_gan.decode_batch(mel_output)
            audio = waveforms.squeeze().numpy()
            audio = (audio * 2**15).astype(np.int16)
            sound = AudioSegment(audio.tobytes(), frame_rate=22050, sample_width=2, channels=1)
            filename = f"{word.replace(' ', '_')}_TTS.wav"
            sound.export(filename, format="wav")
            wave_obj = sa.WaveObject.from_wave_file(filename)
            play_obj = wave_obj.play()
            play_obj.wait_done()
            return filename  # return the filename
        elif self.tts == "ElevenLabs":
            user = ElevenLabsUser(self.tts2_api_key)
            voice = user.get_voices_by_name("Rachel")[0]
            voice.generate_play_audio_v2(text, playbackOptions=PlaybackOptions(runInBackground=False))

    def main(self, words):
        file_paths = []
        for word in words:
            definition, example_sentence = self.get_definition(word)
            if definition is None or example_sentence is None:
                # Skip this word if it was not found
                continue
            text = f'Word: {word}. Definition: {definition}. Example: {example_sentence}'
            file_path = self.text_to_speech(text, word)
            file_paths.append(file_path)
        merge_audio_files(file_paths, "output.mp3")

    def run(self, text):
        words = re.split(',|\n', text)
        words = [word.strip() for word in words if word.strip() != '']
        self.main(words)

class App(QWidget):
    def __init__(self):
        super().__init__()

        self.dict_opt = QComboBox()
        self.dict_opt.addItems(["merriam", "gpt"])

        self.tts_opt = QComboBox()
        self.tts_opt.addItems(["Tacotron2", "ElevenLabs"])

        self.gpu_opt = QComboBox()
        self.gpu_opt.addItems(["GPU", "No GPU"])

        self.tts_api_key = QLineEdit()
        self.merriam_api_key = QLineEdit()
        self.openai_api_key = QLineEdit()
        self.words_input = QTextEdit()

        self.start_button = QPushButton('Start')
        self.start_button.clicked.connect(self.start_processing)

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Dictionary:"))
        layout.addWidget(self.dict_opt)
        layout.addWidget(QLabel("TTS Engine:"))
        layout.addWidget(self.tts_opt)
        layout.addWidget(QLabel("Run on:"))
        layout.addWidget(self.gpu_opt)
        layout.addWidget(QLabel("API Keys (TTS):"))
        layout.addWidget(self.tts_api_key)
        layout.addWidget(QLabel("API Keys (merriam):"))
        layout.addWidget(self.merriam_api_key)
        layout.addWidget(QLabel("API Keys (OpenAI):"))
        layout.addWidget(self.openai_api_key)
        layout.addWidget(QLabel("Enter words:"))
        layout.addWidget(self.words_input)
        layout.addWidget(self.start_button)
        self.setLayout(layout)

    def start_processing(self):
        words = self.words_input.toPlainText()
        tts_engine = self.tts_opt.currentText()
        dictionary = self.dict_opt.currentText()

        if tts_engine == "ElevenLabs":
            tts_api_key = self.tts_api_key.text()
        else:
            tts_api_key = None

        merriam_api_key = self.merriam_api_key.text()
        gpu_opt = self.gpu_opt.currentText()
        openai_api_key = self.openai_api_key.text()
        word_definition_and_tts = WordDefinitionAndTTS(merriam_api_key, 'https://www.dictionaryapi.com/api/v3/references/thesaurus/json/', tts_engine, dictionary, tts_api_key, gpu_opt, openai_api_key)
        word_definition_and_tts.run(words)

app = QApplication([])
window = App()
window.show()
app.exec_()