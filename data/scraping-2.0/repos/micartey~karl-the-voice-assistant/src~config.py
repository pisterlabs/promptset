import os
import sys

from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI

# Load .env file
load_dotenv()

#
# Configure logging
#
logger.add(
    sys.stdout, format="{time} {level} {message}", filter="my_module", level="INFO"
)
logger.add("logs/log_{time}.log", rotation="10 MB", serialize=True)

#
# Environment variables
#
OPENAI_API_TOKEN = os.getenv("OPENAI_API_TOKEN")
WAKE_WORD = os.getenv("WAKE_WORD")
ROLE = os.getenv("ROLE")

AMBIENT_NOISE_LEVEL = os.getenv("AMBIENT_NOISE_LEVEL")
SAMPLE_RATE = os.getenv("SAMPLE_RATE")
CHUNK_SIZE = os.getenv("CHUNK_SIZE")

DEBUG_RMS = os.getenv("DEBUG_RMS")

#
# Objects
#
openai = OpenAI(api_key=OPENAI_API_TOKEN)

abort_prompt = [
    {
        "role": "system",
        "content": "You will be provided with some words or sentences. If the sentences go into "
        'the direction to abort, stop or to be silent: Return "Y". Else Return "N"',
    },
]
