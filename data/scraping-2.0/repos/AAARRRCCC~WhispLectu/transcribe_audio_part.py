import os
import openai

def transcribe_audio_part(audio_file_path):
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OpenAI API key not found in environment variables.")

    with open(audio_file_path, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)

    # Construct the output path for the transcript based on the audio file's path
    transcript_file_name = os.path.basename(audio_file_path).replace(".mp3", "_transcript.txt")
    transcript_file_path = os.path.join(os.path.dirname(audio_file_path), transcript_file_name)
    
    with open(transcript_file_path, "w") as transcript_file:
        transcript_file.write(transcript["text"])

    print("Transcription completed for:", audio_file_path)
