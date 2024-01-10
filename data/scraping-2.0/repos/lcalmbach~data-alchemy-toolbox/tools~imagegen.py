import streamlit as st
import os
from openai import OpenAI
from enum import Enum

from helper import get_image_from_url
from tools.tool_base import ToolBase, TEMP_PATH, OUTPUT_PATH, DEMO_PATH

DEMO_FILE = DEMO_PATH + "000000406031.pdf"
TEMP_FILE = TEMP_PATH + "temp_audio."
OUTPUT_FILE = OUTPUT_PATH + "audio_output.mp3"
FILE_FORMAT_OPTIONS = ["jpg", "jepeg", "png", "gif", "bmp", "tiff", "tif"]


class InputFormat(Enum):
    DEMO = 0


class ImageGenerator(ToolBase):
    def __init__(self, logger):
        super().__init__(logger)
        self.title = "Bildgenerator"
        self.formats = ["Demo"]
        self.input_file = None
        self.output_file = None
        self.prompt = ""
        self.image_url = None

        self.script_name, script_extension = os.path.splitext(__file__)
        self.intro = self.get_intro()

    def show_settings(self):
        self.input_type = st.radio("Input für ImageGen", options=self.formats)
        size_options = ["1024x1024", "1024x1792", "1792x1024"]
        self.size = st.selectbox("Grösse des Bildes (HxB)", options=size_options)
        self.quality = st.radio("Qualität des Bildes", options=["standard", "hd"])
        self.n = st.number_input(
            "Anzahl der Bilder", min_value=1, max_value=10, value=1, step=1
        )
        if self.formats.index(self.input_type) == InputFormat.DEMO.value:
            self.prompt = st.text_area(
                "Beispiel",
                value="Ein Bild von vielen bunten Spiel-Würfeln, die einen Baselerstab (Wappen der Stadt Basel) bilden.",
                height=300,
            )

    def generate_image(self, user_prompt: str) -> str:
        client = OpenAI()
        response = client.images.generate(
            model="dall-e-3",
            prompt=user_prompt,
            size=self.size,
            quality=self.quality,
            n=1,
        )
        self.image_url = response.data[0].url

    def run(self):
        self.text = st.text_area(
            "Beschreibe das gewünschte Bild", value=self.prompt, height=300
        )
        if st.button("Starten"):
            with st.spinner("Generiere Bilder..."):
                self.generate_image(self.text)

        if self.image_url:
            image = get_image_from_url(self.image_url)
            if image:
                st.image(image, caption="Downloaded Image", use_column_width=True)

                # Download Button
                st.download_button(
                    label="Download Image",
                    data=image,
                    file_name="downloaded_image.jpg",
                    mime="image/jpeg",
                )
            else:
                st.error("Failed to download image.")
