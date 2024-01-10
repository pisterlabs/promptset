import openai
import os

from dotenv import load_dotenv
from openai import OpenAI
from IPython.display import Audio
from pathlib import Path

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

print(openai.api_key)

client = OpenAI()

story = """In the image, we see a moment of distress captured along a tranquil suburban roadway. A man sits on the 
curb, his body language speaking volumes as he holds his head in his hands, his elbow resting on his knee. The cause 
of his dismay is immediately apparent: before him stands a vehicle, a pickup truck that has clearly seen better days. 
Its once smooth facade is marred by an encounter of an unfortunate kind; the hood is crumpled, the front bumper 
partially torn off, revealing the inner workings beneath the metallic skin.

The setting is peaceful, lined with lush green trees that dapple the road with patches of shade. Yet, this serenity 
starkly contrasts with the scene of the incident. It isn't hard to imagine what might have transpired—a moment's 
distraction, an unforeseen obstacle, and in the blink of an eye, metal twisted, and a day turned on its head.

In the backdrop, there's another vehicle, but it appears unscathed from this angle, suggesting it might be a passerby 
or perhaps someone stopped to assist. The road is otherwise clear, indicating that any immediate danger has passed 
and what remains now is the aftermath. It's a quiet moment of reflection and recuperation, as the man must now 
wrestle with the ramifications of the event—a reminder of the fragility of our daily routines and the unexpected 
turns life can take."""

response = client.audio.speech.create(
  model="tts-1",
  voice="onyx",
  input=story
)

# Define the path where you want to save the file
speech_file_path = Path('content/story.mp3')

# Save the response content (binary content of the mp3 file) to the path
with open(speech_file_path, 'wb') as file:
    file.write(response.content)

# Play the audio file
Audio(speech_file_path, autoplay=True)