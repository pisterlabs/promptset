from elevenlabs import generate, play
import openai
import speech_recognition as sr
import os
import playsound
from elevenlabs import set_api_key
set_api_key("1234567890")

# Me equivoque


openai.organization = "org-Om6vnsWVREzBoRZGuOnk9XJZ"
openai.api_key = "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"

# Configura el reconocimiento de voz
r = sr.Recognizer()
mic = sr.Microphone(device_index=0)

# Inicia la transcripción de audio en tiempo real desde el micrófono
with mic as source:
    print("Habla ahora...")
    r.adjust_for_ambient_noise(source)  # ajusta el nivel de ruido ambiente
    audio_data = r.record(source, duration=5)  # graba 5 segundos de audio

# guarda el archivo de audio en disco
with open("audio.wav", "wb") as f:
    f.write(audio_data.get_wav_data())

# Transcribe el audio utilizando OpenAI API
with open("audio.wav", "rb") as audio_file:
    transcript = openai.Audio.transcribe(
        "whisper-1", audio_file, language="es")

text = transcript.text

print("TEXTO TRANSCRITO")
print(text)

# Sección de ChatGPT
model_engine = "text-davinci-003"  # elige el modelo que desees utilizar
# model_engine = "davinci-es-002"  # elige el modelo que desees utilizar

question = text

# Genera la respuesta con la API de ChatGPT
response = openai.Completion.create(
    engine=model_engine,
    prompt=(f"{question}"),
    max_tokens=4048
)

# Imprime la respuesta generada
print("RESPUESTA DE CHAT GPT")
salida = response.choices[0].text.strip()
print(salida)

# Elimina el archivo temporal de audio
os.remove("audio.wav")

# seccion para pasar a audio


voice = "Bella"
audio = generate(
    text=f",  {salida}",
    voice=voice
)

play(audio)
