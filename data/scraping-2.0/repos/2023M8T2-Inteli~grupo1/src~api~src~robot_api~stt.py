import sys
from openai import OpenAI
from pathlib import Path
import os
from dotenv import load_dotenv

# Carrega as variáveis de ambiente do arquivo .env
load_dotenv()


api_key = os.getenv("OPENAI_KEY")

client = OpenAI(
  api_key=api_key
)


def main(args=sys.argv):
    audio_file = open(Path(__file__).parent.parent.parent / f"src/controllers/temp/audio.mp3", "rb")
    transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    response_format="text"
    )
    print(transcript)

if __name__ == "__main__":
    main()
