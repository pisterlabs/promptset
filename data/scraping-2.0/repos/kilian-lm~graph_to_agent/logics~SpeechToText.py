# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPEN_AI_KEY')
openai.api_key = os.environ.get('OPEN_AI_KEY')



audio_file= open("../test_mp3/20230712113923_next_tow.mp3", "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file)


transcript
type(transcript)

import re
# Assuming you have the 'transcript' variable defined as in your code snippet
text = transcript["text"]

# Split the text based on common grammatical signs
sentences = re.split(r'[.,!?]', text)

# Optionally, you can remove any extra whitespace
sentences = [sentence.strip() for sentence in sentences]

# 'sentences' now contains a list of separate strings, based on the punctuation
for sentence in sentences:
    print(sentence)

print(transcript)


# Extract transcribed text directly
transcribed_text = transcript["text"]

transcript["text"]

transcribed_text = str(transcript["text"])


# Split transcribed text into responses
responses = transcribed_text.split(' ')


confidence = np.mean([alternative['confidence'] for alternative in transcript['alternatives']])


