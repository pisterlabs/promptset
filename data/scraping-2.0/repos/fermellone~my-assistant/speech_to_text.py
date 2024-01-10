def transcript(file_path: str) -> str:
    import openai, os

    audio_file = open(file_path, "rb")
    transcription = openai.Audio.transcribe(
        model="whisper-1", file=audio_file, api_key=os.getenv("OPENAI_API_KEY")
    )

    return transcription["text"]
