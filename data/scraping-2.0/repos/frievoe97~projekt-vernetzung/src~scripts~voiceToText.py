from moviepy.editor import *
from openai import OpenAI

# Pfad zur Videodatei
video_path = 'temp/interview_01.mp4'

# Pfad zur Ausgabedatei für die Audiospur
audio_output_path = 'audiospur.wav'

# Videodatei öffnen
video_clip = VideoFileClip(video_path)

# Audiospur extrahieren
audio_clip = video_clip.audio

# Audiospur speichern
audio_clip.write_audiofile(audio_output_path)

# Dateien schließen
video_clip.close()
audio_clip.close()

# OpenAI API-Key
api_key = ''

# Datei mit der Audiospur öffnen
audio_file = open(audio_output_path, "rb")

# OpenAI Transkription
client = OpenAI(api_key)
transcript = client.audio.transcriptions.create(
    model="whisper-1",
    file=audio_file,
    response_format="text"
)

# Transkription ausgeben
print(transcript['text'])

# Datei mit der Audiospur schließen
audio_file.close()
