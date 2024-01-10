# Import required libraries:
import openai  # audio to text
from gtts import gTTS  # text to audio

openai.api_key = "<YOUR OPEN-AI API KEY>"

# Transcribe and convert audio to text file
audio_path = "./ds.mp3"
audio_file = open(audio_path, "rb")
transcript = openai.Audio.transcribe("whisper-1", audio_file)
print(transcript.text)

txt_to_audio = transcript.text

# Convert text to audio in French language
tts = gTTS(text=txt_to_audio, lang='fr', slow=False)

# save audio file
tts.save('French.mp3')