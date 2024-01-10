import os
import openai

openai.api_type = "azure"
openai.api_base = "https://neyaheurope.openai.azure.com/"
openai.api_version = "2023-09-01-preview"
openai.api_key = os.getenv("OPENAI_API_KEY")

audio_file=open("aws_lambda.mp3","rb")

response=openai.Audio.transcribe(
  model="whisper-1",
  deployment_id="audio_demo",
  file=audio_file,
  response_format="text"
)

response=openai.ChatCompletion.create(
  engine="text_demo",
  messages=[{"role":"user","content":"Summarize the text in to 5 bullet points: "+response}]
)

print(response)



