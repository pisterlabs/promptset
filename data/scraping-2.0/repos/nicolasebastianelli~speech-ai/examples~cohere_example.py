import os

from dotenv import load_dotenv

from speechai import SpeechAI
from speechai.llm import Cohere
from speechai.tts import GTTS

load_dotenv()
api_key = os.getenv("COHERE-KEY")

cohere = Cohere(api_key)
tts = GTTS(language_code="it")
sa = SpeechAI(cohere, tts)

sa.synthesize_dialog("Quanto sono alte le giraffe in media?", "./output/giraffe-cohere-it.mp3")
[text, audio] = sa.synthesize_dialog("Quale è l'animale più veloce sulla terra?", "./output/fastest-cohere-it.mp3")
print(text, audio)
