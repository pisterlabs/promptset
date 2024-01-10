import streamlit as st
import numpy as np
import cv2
from PIL import Image
import pytesseract
import openai
from gtts import gTTS
import os
import sounddevice as sd
import soundfile as sf


pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


openai.api_key = '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'

def main():
    st.title("QUOTE GURU APP")
    image=Image.open(r"C:\Users\SREERAJ\Downloads\deadpoetssociety.jpg")
    st.image(image)

    uploaded_image = st.file_uploader("     UPLOAD AN IMAGE    ", type=["jpg", "png"])

    if uploaded_image:
        img = Image.open(uploaded_image)
        img_array = np.array(img)
        
        text = pytesseract.image_to_string(img_array)
        text = text.replace("\n", "")

        prompt = f"Please provide a detailed explanation of the meaning of the following quote: '{text}'"
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=1000,
            temperature=0.1,
        )
        generated_translation = response.choices[0].text.strip()

        st.subheader("    GENERATED TEXT:   ")
        st.write(generated_translation)

        tts = gTTS(text=generated_translation, lang="en", slow=False)
        tts.save("output.mp3")
        audio_file = "output.mp3"

        data, sample_rate = sf.read(audio_file)

        st.subheader("  GENERATED AUDIO:  ")
        st.audio(data, format="audio/mp3", sample_rate=sample_rate) 

if __name__ == "__main__":
    main()
