from openai import OpenAI
client = OpenAI(api_key="")

def speech_to_text(audio_file_path):
  audio_file = open(audio_file_path, "rb")
  transcript = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file, 
    response_format="text"
  )
  return transcript
