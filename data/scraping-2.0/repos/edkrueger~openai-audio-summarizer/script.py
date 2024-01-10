import os

from dotenv import load_dotenv
import openai
from pydub import AudioSegment

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

IN_PATH = "data/otm070723_cms1337550_pod.mp3_ywr3ahjkcgo_2547cfdb20b60f40eb57563221685074_49570906.mp3"
SLICE_PATH = "data/hinton.mp3"
TRANSCRIPT_PATH = "data/hinton_transcript.txt"
SUMMARIZATION_PROMPT = """
You are an API that summarizes text. When you receive text summarize it.

Emphasize Hinton's points.
Emphasize points about how AI's mind is similar to human minds.
Don't emphasize dangers of AI.

text: {}
"""

# starts at 16:55
START_MINUTES_PART = 16
START_SECONDS_PART = 55
START_MS = ((START_MINUTES_PART * 60) + START_SECONDS_PART) * 1000

# ends at 35:08
END_MINUTES = 35
END_SECONDS = 8
END_MS = ((END_MINUTES * 60) + END_SECONDS) * 1000

openai.api_key = OPENAI_API_KEY

podcast = AudioSegment.from_mp3(IN_PATH)
interview = podcast[START_MS: END_MS]

interview.export(SLICE_PATH, format="mp3")

with open(SLICE_PATH, "rb") as file:
    transcript = openai.Audio.transcribe("whisper-1", file)

text = transcript["text"]

with open(TRANSCRIPT_PATH, "w+") as file:
    file.write(text)

# create a chat completion
chat_completion = openai.ChatCompletion.create(model="gpt-4", messages=[{"role": "user", "content": SUMMARIZATION_PROMPT.format(text)}])

# print the chat completion
print(chat_completion.choices[0].message.content)







