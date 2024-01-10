import streamlit as st
import os
from moviepy.editor import VideoFileClip
from openai import OpenAI
import pyperclip
from enum import Enum
import zipfile

from helper import get_var, save_uploadedfile
from tools.tool_base import ToolBase, TEMP_PATH, OUTPUT_PATH, DEMO_PATH

DEMO_FILE = DEMO_PATH + "demo_audio.mp3"
TEMP_FILE = TEMP_PATH + "temp_audio."
OUTPUT_FILE = OUTPUT_PATH + "audio_output.txt"
FILE_FORMAT_OPTIONS = ["mp4", "mp3"]


class InputFormat(Enum):
    DEMO = 0
    FILE = 1
    ZIPPED_FILE = 2
    S3 = 3


class Speech2Text(ToolBase):
    def __init__(self, logger):
        super().__init__(logger)
        self.title = "Audio zu Text"
        self.formats = ["Demo", "Audio/Video Datei", "Sammlung von Audio-Dateien (zip)"]
        self.script_name, script_extension = os.path.splitext(__file__)
        self.intro = self.get_intro()
        self.input_file = None
        self.output_file = None
        self.text = ""

    def show_settings(self):
        """
        Zeigt die Einstellungen für Speech2Text an.

        Diese Methode ermöglicht es dem Benutzer, den Eingabeformaten für Speech2Text auszuwählen und die entsprechenden Aktionen
        basierend auf dem ausgewählten Eingabetyp zu verarbeiten. Eingabetypen sind Demo (fixed demo Datei), User lädt mp3/4 Datei
        hoch und: User lädt gezippte Datei mit mp3, mp4 Dateien hoch.

        Returns:
            None
        """
        self.input_type = st.radio("Input für Speech2Text", options=self.formats)
        if self.formats.index(self.input_type) == InputFormat.DEMO.value:
            self.output_file = DEMO_FILE
            st.audio(DEMO_FILE)
        elif self.formats.index(self.input_type) == InputFormat.FILE.value:
            self.input_file = st.file_uploader(
                "MP4 oder MP3 Datei hochladen",
                type=FILE_FORMAT_OPTIONS,
                help="Lade die Datei hoch, die du transkribieren möchtest.",
            )
            if self.input_file is not None:
                file = TEMP_PATH + self.input_file.name
                ok, err_msg = save_uploadedfile(self.input_file, TEMP_PATH)
                if ok:
                    self.output_file = file
                    st.audio(self.output_file)
        elif self.formats.index(self.input_type) == InputFormat.ZIPPED_FILE.value:
            self.input_file = st.file_uploader(
                "ZIP Datei mit gezippten MP4 oder MP3 Dateien hochladen",
                type=["zip"],
                help="Lade die ZIP Datei hoch, welche die zu transkribierenden Dateien enthält. Achtung, die Datei darf nicht grösser als 200 MB sein.",
            )

    def extract_audio_from_video(self, video_file: str) -> str:
        audio_file_name = video_file.replace(".mp4", ".mp3")
        video = VideoFileClip(video_file)
        audio = video.audio
        audio.write_audiofile(audio_file_name)
        return audio_file_name

    def transcribe(self, filename: str) -> str:
        if filename.endswith(".mp4"):
            audio_file_name = self.extract_audio_from_video(filename)
        else:
            audio_file_name = filename

        audio_file = open(audio_file_name, "rb")
        client = OpenAI(
            api_key=get_var("OPENAI_API_KEY"),
        )
        transcript = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file, response_format="text"
        )
        return transcript

    def run(self):
        if st.button("Transkribieren"):
            with st.spinner("Transkribiere Audio..."):
                if self.formats.index(self.input_type) == 0:
                    self.text = self.transcribe(self.output_file)
                elif self.formats.index(self.input_type) == InputFormat.FILE.value:
                    self.text = self.transcribe(self.output_file)
                elif (
                    self.formats.index(self.input_type) == InputFormat.ZIPPED_FILE.value
                ):
                    with zipfile.ZipFile(self.input_file, "r") as zip_ref:
                        self.output_file = os.path.join(OUTPUT_PATH, "transcribed.zip")
                        with zipfile.ZipFile(self.output_file, "w") as out_zip:
                            for file_name in zip_ref.namelist():
                                zip_ref.extract(file_name, TEMP_PATH)
                                transcribed_text = self.transcribe(
                                    os.path.join(TEMP_PATH, file_name)
                                )
                                txt_filename = os.path.splitext(file_name)[0] + ".txt"
                                txt_path = os.path.join(TEMP_PATH, txt_filename)
                                with open(txt_path, "w") as txt_file:
                                    txt_file.write(transcribed_text)
                                out_zip.write(txt_path, arcname=txt_filename)
                    st.success("Transkription abgeschlossen.")
                else:
                    st.info("Diese Option ist noch nicht verfügbar.")
        if self.text != "":
            st.markdown("**Transkript**")
            st.markdown(self.text)
            cols = st.columns(2, gap="small")
            with cols[0]:
                if st.button("📋 Text in Zwischenablage kopieren"):
                    pyperclip.copy(
                        self.text,
                    )
            with cols[1]:
                st.download_button(
                    label="⬇️ Datei herunterladen",
                    data=self.text,
                    file_name=OUTPUT_FILE,
                    mime="text/plain",
                )
        elif (
            self.formats.index(self.input_type) == InputFormat.ZIPPED_FILE.value
            and self.output_file
        ):
            st.download_button(
                label="⬇️ Transkripte herunterladen",
                data=self.text,
                file_name=self.output_file,
                mime="application/zip",
            )
