from pathlib import Path
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def criar_arquivo_de_fala(modelo="tts-1", voz="alloy", entrada=None, nome_arquivo="original.mp3"):
    if entrada is None:
        raise ValueError("A entrada não pode ser None.")

    speech_file_path = Path(__file__).parent / nome_arquivo

    response = client.audio.speech.create(
        model=modelo,
        voice=voz,
        input=entrada
    )

    print("Audio criado com sucesso!")
    response.stream_to_file(speech_file_path)

if __name__ == "__main__":
    criar_arquivo_de_fala(entrada="Olá, essa é a atividade 8 da Manu!")
