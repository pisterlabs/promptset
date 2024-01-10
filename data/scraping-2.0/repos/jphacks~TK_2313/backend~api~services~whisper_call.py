import openai


def whisper_transcription(audio_file: bytes):
    transcript = openai.Audio.transcribe_raw(
        "whisper-1", audio_file, "input.wav")
    return transcript['text']
