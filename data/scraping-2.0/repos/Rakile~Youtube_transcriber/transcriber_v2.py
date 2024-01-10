import asyncio
import os
import subprocess
import sys
import threading

from boilerplate import API
from PySide6.QtCore import QMetaObject, Qt, Slot, QUrl, QSize
from PySide6.QtMultimedia import QMediaPlayer, QAudioOutput
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QPushButton, QTextEdit, QComboBox


from pytube import YouTube
import whisper
import time
import shutil

from openai import OpenAI

client = None
novelToken = None

class YouTubeTranscriber(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.currentModel = ""
        self.player = QMediaPlayer()
    def initUI(self):
        self.resize(QSize(600,600))
        # Layout
        layout = QVBoxLayout(self)


        # Novel AI persisten api token
        self.novelai_persistent_api_key = QLineEdit(self)
        self.novelai_persistent_api_key.setPlaceholderText("<Enter NovelAI Persistent API Token here. Leave blank if you don't have>")
        layout.addWidget(self.novelai_persistent_api_key)

        # Novel AI persisten api token
        self.openai_persistent_api_key = QLineEdit(self)
        self.openai_persistent_api_key.setPlaceholderText("<Enter OpenAI API Key here. Leave blank if you don't have.>")
        layout.addWidget(self.openai_persistent_api_key)

        # YouTube URL input
        self.url_input = QLineEdit(self)
        self.url_input.setPlaceholderText("<Enter YouTube URL here, or path to video or audio file.>")
        layout.addWidget(self.url_input)


        # Whisper Model Selection Dropdown
        self.model_selection = QComboBox(self)
        self.model_selection.addItems(["tiny", "small", "base", "medium", "large"])
        layout.addWidget(self.model_selection)

        #Use engine
        # Whisper Model Selection Dropdown
        self.engine_selection = QComboBox(self)
        self.engine_selection.addItems(["OpenAI", "NovelAI"])
        layout.addWidget(self.engine_selection)
        self.engine_selection.currentTextChanged.connect(self.update_voice_selection)

        # New Dropdown for Voice Selection OpenAI
        self.voice_selection = QComboBox(self)
        self.voice_selection.addItems(["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
        layout.addWidget(self.voice_selection)

        # Download and Transcribe Button
        self.download_button = QPushButton("Download and Transcribe", self)
        self.download_button.clicked.connect(self.download_and_transcribe)
        layout.addWidget(self.download_button)

        # Transcription result display
        self.transcription_display = QTextEdit(self)
        self.transcription_display.setReadOnly(True)
        layout.addWidget(self.transcription_display)

        # Stop audio
        self.stopButton = QPushButton("Stop Audio", self)
        self.stopButton.clicked.connect(self.stopAudio)
        layout.addWidget(self.stopButton)

        self.setLayout(layout)
        self.setWindowTitle("YouTube Transcriber")

    def update_voice_selection(self, engine_name):
        if engine_name == "NovelAI":
            self.voice_selection.clear()
            self.voice_selection.addItems(["Ligeia", "Aini", "Orea", "Claea", "Lim", "Aurae", "Naia", "Aulon", "Elei", "Ogma", "Raid", "Pega", "Lam"])
        elif engine_name == "OpenAI":
            self.voice_selection.clear()
            self.voice_selection.addItems(["alloy", "echo", "fable", "onyx", "nova", "shimmer"])

    def get_thread_id(thread):
        # Returns the thread ID
        if hasattr(thread, "_thread_id"):
            return thread._thread_id
        for id, t in threading._active.items():
            if t is thread:
                return id

    def stopAudio(self):
        self.player.stop()
        self.player.deleteLater()
        self.player = QMediaPlayer()

    def download_and_transcribe(self):
        self.liveview_t = threading.Thread(target=self.downloadandtrans)
        self.liveview_t.start()

    def downloadandtrans(self):
        global client
        global novelToken
        if client == None:
            if self.openai_persistent_api_key.text() != "":
                client = OpenAI(api_key=self.openai_persistent_api_key.text())
            else:
                client = None
        elif self.openai_persistent_api_key.text() == "":
            client = None

        if novelToken == None:
            if self.novelai_persistent_api_key.text() != "":
                novelToken = self.novelai_persistent_api_key.text()
            else:
                novelToken = None
        elif self.novelai_persistent_api_key.text() == "":
            novelToken = None


        video_url = self.url_input.text()
        selected_model = self.model_selection.currentText()
        if video_url:
            if video_url.startswith("http"):
                yt = YouTube(video_url)
                audio_stream = yt.streams.filter(only_audio=True).first()
                audio_stream.download(filename='recording.mp3')
            else:
                shutil.copyfile(video_url, 'recording.mp3')

            modelName = self.model_selection.currentText()
            if modelName != self.currentModel:
                print("Loading model " + modelName + ".")
                start = time.time()
                self.model = whisper.load_model(modelName)
                end = time.time()
                print("Loading the model " + modelName + " took:" + str(end - start) + " seconds")
                self.currentModel = modelName

            start = time.time()
            # load audio and pad/trim it to fit 30 seconds
            audio = whisper.load_audio("recording.mp3")
            audio = whisper.pad_or_trim(audio)
            # make log-Mel spectrogram and move to the same device as the model
            if modelName == "large":
                mel = whisper.log_mel_spectrogram(audio, n_mels=128).to(self.model.device)
            else:
                mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

            # detect the spoken language
            _, probs = self.model.detect_language(mel)
            lang = max(probs, key=probs.get)
            print(f"Detected language: {max(probs, key=probs.get)}")

            result = self.model.transcribe("recording.mp3", language=lang, task="translate", fp16=False)
            # result = model.transcribe("recording.mp3", fp16=False)
            # make log-Mel spectrogram and move to the same device as the model
            mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

            end = time.time()
            print("The translation took:" + str(end - start) + " seconds")
            print(f'The text: \n {result["text"]}')
            self.current_text = result["text"]

            QMetaObject.invokeMethod(self, "update_text", Qt.QueuedConnection)

    @Slot()
    def update_text(self):
        self.transcription_display.setText(self.current_text)
        self.liveaudio = threading.Thread(target=self.playAudio)
        self.liveaudio.start()

    async def playAudioNovelAIAsync(self):
        chunks = [self.current_text[i:i + 999] for i in range(0, len(self.current_text), 999)]
        print("Number of chunks to make into audio:" + str(len(chunks)))
        final_audio_file = "output.mp3"
        for index, chunk in enumerate(chunks):
            print("-- Processing chunk:" + str(index))
            # Generate audio for each chunk
            async with API(novelToken) as api_handler:
                api = api_handler.api
                # encryption_key = api_handler.encryption_key
                logger = api_handler.logger
                text = chunk
                voice = self.voice_selection.currentText()
                seed = -1
                # opus = False
                opus = False
                # version = "v1"
                version = "v2"
                #logger.info(f"Generating a tts voice for {len(text)} characters of text")
                tts = await api.low_level.generate_voice(text, voice, seed, opus, version)

            #logger.info(f"TTS saved in {tts_file}")
            # Save each audio chunk to a file
            chunk_file = f"output_chunk_{index}.mp3"
            #chunk_convert = f"output_chunk_{index}.mp3"
            print("Writing chunk to:" + str(chunk_file))
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
            with open(chunk_file, "wb") as f:
                f.write(tts)
            f.close()


            # Concatenate audio files
            if index == 0:
                if os.path.exists(final_audio_file):
                    os.remove(final_audio_file)
                os.rename(chunk_file, final_audio_file)
            else:
                if os.path.exists("temp.mp3"):
                    os.remove("temp.mp3")

                cmd = ["ffmpeg", "-i", f"concat:{final_audio_file}|{chunk_file}", "-acodec", "copy", "temp.mp3", "-map_metadata", "0:1"]
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                process.wait()
                print("Stiched piece " + str(index) + " to the whole...")
                stdout, stderr = process.communicate()
                if os.path.exists(final_audio_file):
                    os.remove(final_audio_file)
                os.rename("temp.mp3", final_audio_file)
                os.remove(chunk_file)

        if os.path.exists("temp.mp3"):
            os.remove("temp.mp3")
        # Play the final concatenated audio file using QMediaPlayer from PySide6
        url = final_audio_file
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.player.setSource(url)
        self.audio_output.setVolume(50)
        self.player.play()

    def playAudio(self):
        currentEngine = self.engine_selection.currentText()
        if currentEngine == "OpenAI":
            if client != None:
                self.playAudioOpenAI()
        else:
            if novelToken != None:
                asyncio.run(self.playAudioNovelAIAsync())


    def playAudioOpenAI(self):
        # Split the text into chunks of 4096 characters
        chunks = [self.current_text[i:i + 4095] for i in range(0, len(self.current_text), 4095)]
        print("Number of chunks to make into audio:" + str(len(chunks)))
        final_audio_file = "output.mp3"

        # Process and concatenate each chunk
        for index, chunk in enumerate(chunks):
            print("-- Processing chunk:" + str(index))
            print("----------------------------------")
            print(str(chunk))
            print("----------------------------------")
            # Generate audio for each chunk
            response = client.audio.speech.create(
                model="tts-1",
                voice=self.voice_selection.currentText(),
                input=chunk,
            )

            # Save each audio chunk to a file
            chunk_file = f"output_chunk_{index}.mp3"
            if os.path.exists(chunk_file):
                os.remove(chunk_file)
            response.stream_to_file(chunk_file)

            # Concatenate audio files
            if index == 0:
                if os.path.exists(final_audio_file):
                    os.remove(final_audio_file)
                os.rename(chunk_file, final_audio_file)
            else:
                if os.path.exists("temp.mp3"):
                    os.remove("temp.mp3")

                cmd = ["ffmpeg", "-i", f"concat:{final_audio_file}|{chunk_file}", "-acodec", "copy", "temp.mp3", "-map_metadata", "0:1"]
                process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
                print("Stiched piece " + str(index) + " to the whole...")
                stdout, stderr = process.communicate()
                if os.path.exists(final_audio_file):
                    os.remove(final_audio_file)
                os.rename("temp.mp3", final_audio_file)
                os.remove(chunk_file)


        if os.path.exists("temp.mp3"):
            os.remove("temp.mp3")


        # Play the final concatenated audio file using QMediaPlayer from PySide6
        url = final_audio_file
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.player.setSource(url)
        self.audio_output.setVolume(50)
        self.player.play()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = YouTubeTranscriber()
    window.show()
    sys.exit(app.exec())