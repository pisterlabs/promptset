import os
import openai


def main():
    key = os.environ.get("WHISPER_API_KEY")
    audio = os.environ.get("WHISPER_FILE")
    file_name = os.environ.get("WHISPER_DESTINATION_FILE_NAME")
    lang = os.environ.get("WHISPER_LANGUAGE")

    openai.api_key = key
    audio_file = open(audio, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file, language=lang)

    with open(file_name, "w") as f:
        f.write(transcript.text)


if __name__ == "__main__":
    main()
