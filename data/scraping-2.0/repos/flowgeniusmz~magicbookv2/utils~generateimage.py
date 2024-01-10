import streamlit as st
from openai import OpenAI
from config import toastalerts as tst

client = OpenAI(api_key=st.secrets.openai.api_key)

def generate_image(varPrompt):
    tst.toast_alert_start("Generating Image...")
    image = client.images.generate(
        model=st.secrets.openai.model_dalle3,
        prompt=varPrompt,
        n=1,
        size = "512x512"
    )

    image_url = image.data[0].url
    tst.toast_alert_start("Image Generated!")
    return image_url