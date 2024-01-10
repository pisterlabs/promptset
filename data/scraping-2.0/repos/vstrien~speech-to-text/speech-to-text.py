import openai
from pydub import AudioSegment
import os
import json

input_filename = "heilige-grond.mp3"
basename = os.path.splitext(input_filename)[0]
extension = os.path.splitext(input_filename)[1]

total_audio = AudioSegment.from_mp3(input_filename)

# Break the audio in 10 minute segments
ten_min = 10 * 60 * 1000
audio_segments = []
for i in range(0, len(total_audio), ten_min):
    audio_segments.append(total_audio[i:i+ten_min])

# Save the segments to disk
for i, segment in enumerate(audio_segments):
    segment.export(f"{basename}-{i}{extension}", format=f"mp3")

# Transcribe everything
for i, _ in enumerate(audio_segments):
    with open(f"{basename}-{i}{extension}", "rb") as audio_file:
        with open(f"{basename}-{i}-output.txt", "w") as f:
            result = json.loads(str(openai.Audio.transcribe("whisper-1", audio_file)))["text"]
            f.write(result)

# Combine all the transcriptions
with open(f"{basename}-output.txt", "w") as f:
    for i, _ in enumerate(audio_segments):
        with open(f"{basename}-{i}-output.txt", "r") as g:
            f.write(g.read())

# Delete mp3 segments
for i, _ in enumerate(audio_segments):
    os.remove(f"{basename}-{i}{extension}")
    os.remove(f"{basename}-{i}-output.txt")