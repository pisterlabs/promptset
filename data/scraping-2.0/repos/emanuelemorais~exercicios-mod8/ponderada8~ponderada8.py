from pathlib import Path
from openai import OpenAI
import os

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Transcreve de audio para texto
def transcrever_audio(nome_arquivo="original.mp3", modelo="whisper-1", formato_resposta="text"):
    with open(nome_arquivo, "rb") as arquivo_audio:
        transcript = client.audio.transcriptions.create(
            model=modelo,
            file=arquivo_audio,
            response_format=formato_resposta
        )
    print(f"O audio possui o seguinte texto: {transcript}")
    return transcript

# Traduz o texto para uma determinada lingua através do chatgpt
def chat_with_gpt(prompt, lingua):

    response = client.chat.completions.create(model="gpt-3.5-turbo", 
    messages=[
        {"role": "system", "content": f"Você é tradutor de texto e deve traduzir para {lingua}"},
        {"role": "user", "content": prompt},
    ])

    obj = response.choices[0].message

    print(f"O texto traduzido para {lingua}: {obj.content}")
    return obj.content

# Menu para escolha de qual lingua será traduzizo
def menu(transcript):
    print("Para qual lingua você deseja traduzir:")
    print("1. Inglês")
    print("2. Japonês")
    print("3. Espanhol")

    dict = {
        "1" : "Inglês",
        "2" : "Janponês",
        "3" : "Espanhol"
    }

    while True:
        escolha = input("Digite o número da opção desejada: ")

        if escolha in ['1', '2', '3']:
            resposta = chat_with_gpt(transcript,dict[escolha])
            return resposta
        else:
            print("Opção inválida. Por favor, escolha uma opção válida.")

# Converte texto para audio
def criar_arquivo_de_fala(modelo="tts-1", voz="alloy", entrada=None, nome_arquivo="speech.mp3"):
    if entrada is None:
        raise ValueError("A entrada não pode ser None.")

    speech_file_path = Path(__file__).parent / nome_arquivo

    response = client.audio.speech.create(
        model=modelo,
        voice=voz,
        input=entrada
    )

    print('Audio finalizado :)') 
    response.stream_to_file(speech_file_path)


if __name__ == "__main__":
  transcript = transcrever_audio()
  resposta_ingles = menu(transcript)
  criar_arquivo_de_fala(entrada=resposta_ingles)
