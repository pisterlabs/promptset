from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI()

if __name__ == "__main__":
    with open("../data/recording-turkish.mp3", "rb") as f:
        translation = client.audio.translations.create(model="whisper-1", file=f)

    print(translation.text)
