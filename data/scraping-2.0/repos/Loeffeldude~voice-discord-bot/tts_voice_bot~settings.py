from dotenv import load_dotenv

load_dotenv()

import os
import openai

# --- OPENAI --- #
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY


# --- DISCORD --- #
DISCORD_TOKEN = os.environ.get("DISCORD_TOKEN")

TALK_DISCONNECT_DELAY = 2
# How many messages to fetch when creating a chat
THREAD_MESSAGE_LIMIT = 100

# --- ELEVEN LABS --- #
VOICE_ID = os.environ.get("VOICE_ID")
ELEVEN_LABS_BASE_URL = "https://api.elevenlabs.io"
STABILITY = 0.2
SIMULARITY_BOOST = 0.7
