from openai import OpenAI
client = OpenAI(api_key="<insert OpenAI API key>")

audio_file = open("<your audio file>", "rb")
transcript = client.audio.translations.create(
  model="whisper-1", 
  file=audio_file, 
  response_format="text"
)
print(transcript)
