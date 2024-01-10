from langchain.llms import Ollama
from gtts import gTTS
from io import BytesIO
import pygame

ollama = Ollama(base_url='http://localhost:11434',
model="ajudante")

pygame.mixer.init()

def speak(text, language):
    # Objeto BytesIO
    audio = BytesIO()
    # Salva o audio no objeto BytesIO
    audio_confg = gTTS(text=text, lang=language)
    audio_confg.write_to_fp(audio)
    audio.seek(0)
    pygame.mixer.music.load(audio)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

print(ollama('quem é você?'))