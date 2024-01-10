import os
import openai


def main():
    key = os.environ.get("WHISPER_API_KEY")
    audio = os.environ.get("WHISPER_FILE")
    export_file = os.environ.get("WHISPER_DESTINATION_FILE_NAME")

    openai.api_key = key
    audio_file = open(audio, "rb")
    transcript = openai.Audio.translate("whisper-1", audio_file)

    with open(export_file, "w") as f:
        f.write(transcript.text)


if __name__ == "__main__":
    main()
