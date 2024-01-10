import os
import random

import openai
import requests
import streamlit as st

from utils.logging import configure_logger

logger = configure_logger(__file__)

LOGOS_DIR = "assets/logos/"


def generate_new_logo():
    image = openai.Image.create(
        prompt="Picture a photo, showing a pondering robot with a lightbulb over its head. "
        "It is wearing glasses, and is reading a book. "
        "It has a mustache, and looks like a philosopher.",
        n=1,
        size="512x512",
        api_key=st.secrets.openai_api.key,
    )
    response = requests.get(image["data"][0]["url"])
    if response.ok:
        idx = max(int(f.split("_")[-1].split(".")[0]) for f in os.listdir(LOGOS_DIR))
        logo_path = os.path.join(
            LOGOS_DIR, f"generated_logo_{str(idx + 1).zfill(2)}.png"
        )
        with open(logo_path, "wb") as f:
            f.write(response.content)
    return logo_path


def pick_random_logo():
    if logo_files := os.listdir(LOGOS_DIR):
        return os.path.join(LOGOS_DIR, random.choice(logo_files))


@st.cache_resource(show_spinner="Generating logo...")
def new_logo():
    if st.session_state.setdefault("first_logo", True):
        st.session_state.first_logo = False
        return pick_random_logo()
    probability = 1 / 10
    return pick_random_logo() if random.random() > probability else generate_new_logo()
