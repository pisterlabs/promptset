from openai import OpenAI
import requests
from dotenv import load_dotenv
import os

client = OpenAI()

def audio_to_text():
    audio_file= open("audio/videoplayback.mp3", "rb")
    transcript = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file,
    language="de",
    response_format="text"
    )
    return transcript

def openai_text_to_speech(api_key, input_text, voice="alloy", output_file="speech.mp3"):
    url = "https://api.openai.com/v1/audio/speech"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "tts-1",
        "input": input_text,
        "voice": voice
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        with open(output_file, "wb") as output_audio:
            output_audio.write(response.content)
        print(f"Speech saved to {output_file}")
    else:
        print(f"Error {response.status_code}: {response.text}")

def main():
    load_dotenv()  # Carrega as variáveis de ambiente do arquivo .env
    api_key = os.getenv("OPENAI_API_KEY")  # Obtém o valor da variável de ambiente
    if not api_key:
        raise ValueError("OPENAI_API_KEY não encontrado no arquivo .env")

    input_text = audio_to_text()
    print(input_text)

    openai_text_to_speech(api_key, input_text)

if __name__ == "__main__":
    main()