import streamlit as st
import os
from openai import OpenAI
from enum import Enum

from helper import get_var, extract_text_from_file, save_uploadedfile
from tools.tool_base import ToolBase, TEMP_PATH, OUTPUT_PATH, DEMO_PATH

DEMO_FILE = DEMO_PATH + "000000406031.pdf"
TEMP_FILE = TEMP_PATH + "temp_audio."
OUTPUT_FILE = OUTPUT_PATH + "audio_output.mp3"
FILE_FORMAT_OPTIONS = ["mp4", "mp3"]


class InputFormat(Enum):
    DEMO = 0


class Text2Speech(ToolBase):
    def __init__(self, logger):
        super().__init__(logger)
        self.title = "Text zu Audio"
        self.formats = ["Demo"]
        self.input_file = None
        self.output_file = None
        self.text = ""

        self.script_name, script_extension = os.path.splitext(__file__)
        self.intro = self.get_intro()

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
            text = extract_text_from_file(DEMO_FILE)
            with st.expander("Text"):
                self.text = st.text_area(
                    "Text", value=text, label_visibility="hidden", height=300
                )
                st.markdown(
                    "Du kannst diesen Text bearbeiten oder ersetzen und dann konvertieren."
                )
        else:
            st.info("Diese Option ist noch nicht verfügbar.")

    def convert2audio(self, text: str) -> str:
        client = OpenAI()
        response = client.audio.speech.create(model="tts-1", voice="alloy", input=text)
        response.stream_to_file(OUTPUT_FILE)

    def run(self):
        if st.button("Konvertiere"):
            with st.spinner("Konvertiere Text zu Audio..."):
                self.convert2audio(self.text)

        st.write(OUTPUT_FILE.replace(OUTPUT_PATH, ""))
        if os.path.exists(OUTPUT_FILE):
            st.audio(OUTPUT_FILE)
            with open(OUTPUT_FILE, "rb") as file:
                btn = st.download_button(
                    label="⬇️ Datei herunterladen",
                    data=file,
                    file_name=OUTPUT_FILE.replace(OUTPUT_PATH, ""),
                    mime="audio/mpeg",
                )
