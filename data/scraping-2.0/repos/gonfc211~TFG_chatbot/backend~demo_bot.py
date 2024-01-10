import openai
from dotenv import load_dotenv
import os


load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

context = [

    {'role': 'system', 'content': 'Eres un asistente para dar soporte y resolver dudas a estudiantes sobre cursos de la plataforma Open edX'}

]


def get_completion_from_messages(messages):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
    )

    return response.choices[0].message["content"]


def get_transcritpion_from_message(audio):

    transcript = openai.Audio.transcribe("whisper-1", audio)

    print(transcript)

    return transcript['text'].encode('latin1').decode('unicode_escape')

