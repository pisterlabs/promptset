#!/usr/bin/env python3
#
# get-text.py

import os
import sys
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

file = sys.argv[1]

audio_file = open(file, "rb")
transcript = client.audio.transcribe("whisper-1", audio_file)

print(transcript)

