import os
import openai
import sys
openai.api_key = os.getenv("OPENAI_API_KEY")

if len(sys.argv) < 3:
    print("Usage: python whisper.py <filename> <outfile>")
    sys.exit(1)

filename = sys.argv[1]

outfile  = sys.argv[2]

#print(filename)

audio_file = open(filename, "rb")

print("Transcribing audio file: " + filename)

transcript = openai.Audio.transcribe("whisper-1", audio_file)

audio_file.close()

#print(transcript)

transcript_text = transcript["text"]

#transcript_text = transcript_text.decode("utf-8")

print(transcript_text)

# write the transcript text to a file
with open(outfile, "w") as f:
    f.write(transcript_text)


