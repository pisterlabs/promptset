from openai import OpenAI
import whisper
import soundfile as sf
from videoproc import youtube_preprocess
from chunking import chunk_by_size

# Load the Whisper model once, outside the loop
model = whisper.load_model("tiny")

# Process the YouTube video
audio_file = youtube_preprocess("https://youtu.be/HQjjgi6271k?si=tPJRuetEMA603s-a")

# Determine the number of chunks
no_of_chunks = chunk_by_size(audio_file)

# Process each chunk
for i in range(no_of_chunks):
    file_path = f"process_chunks/chunk{i}.wav"
    print(file_path)

    # Read the audio file
    audio, samplerate = sf.read(file_path)

    # Transcribe the audio
    transcript = model.transcribe(audio)

    # Append the transcription to a file
    with open("videotext.txt", "a") as f:
        f.write(transcript["text"] + "\n")
