from pathlib import Path

from openai import OpenAI

openai = OpenAI()
# speech_file_path = Path(__file__).parent / "speech.mp3"
speech_file_path = "speech.mp3"


def main() -> None:
    audio = openai.audio
    
    # Create text-to-speech audio file
    response = audio.speech.create(
        model="tts-1", voice="nova", input="Good morning! Hello! Goodbye!", speed=0.7,
    )

    response.stream_to_file(speech_file_path)


if __name__ == "__main__":
    main()
