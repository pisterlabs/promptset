from pytube import YouTube
from pydub import AudioSegment
import openai

def audio2text(api_key, audio_file, text_file):
    # read audio file
    audio_file = AudioSegment.from_file(audio_file, format="m4a")

    # cut audio file into chunks
    chunk_size = 100 * 1000  # 100 秒
    chunks = [audio_file[i:i+chunk_size] for i in range(0, len(audio_file), chunk_size)]

    # transcribe and combine chunks
    openai.api_key = api_key
    transcript = ""
    count = 0
    for chunk in chunks:
        with chunk.export("./temp_files/temp.wav", format="wav") as f:
            result = openai.Audio.transcribe("whisper-1", f)
            transcript += result["text"]
        count += 1
        if count == 100:
            break

    with open(text_file, "w") as f:
        f.write(transcript)
    # print(transcript)

