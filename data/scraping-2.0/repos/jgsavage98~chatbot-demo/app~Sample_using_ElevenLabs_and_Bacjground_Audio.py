
import requests
from pydub import AudioSegment
import openai
from io import BytesIO
from azure.storage.blob import BlobServiceClient

openai.api_key ='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'
client_name = "Toby"
CHUNK_SIZE = 1024
# url = "https://api.elevenlabs.io/v1/text-to-speech/777yuCuW9suIZEWPrrI7"

# This is the path to Fiona's voice on ElevenLabs
url = "https://api.elevenlabs.io/v1/text-to-speech/7L2F0M8ojXkZ6StuH3Zr/stream?optimize_streaming_latency=4"

# This is the header to access Fiona's voice on ElevenLabs
headers = {
  "Accept": "audio/mpeg",
  "Content-Type": "application/json",
  "xi-api-key": "0c52b0aafc66e364c2338723afba2a22" #Fiona's API key
  # "xi-api-key": "4e70d3c60c3478c19d8017048d7273f6" #John's API key
}

# This is the OpenAI Prompt, which will likely be modified with Fiona's feedbck
prompt = f"""You are a guided mediation expert. Your client, {client_name}, is struggling with finding a job. He wants to release self doubt, worrying about the future and the unknown. He'd like to feel self assured moving forward. He'd like to feel supported going into the unknown and dealing with change. He just finished grad school. He'd like to meditate for 5 minutes. 
Write a personalized meditation script for {client_name} without any introduction, titles, or headings. Use his name in the script to make it more personal. Add in the following string : "<break time=\"1.0s\" />" after every sentence, and this string: "<break time=\"3.0s\" />" wherever you think a 3 second pause is appropriate. Make the whole meditation 5 minutes long. Do not include any titles or headings."""

# Generate meditation script using OpenAI
response = openai.Completion.create(
    engine="text-davinci-003",
    prompt=f"{prompt}",
    max_tokens=1000
)
script = response.choices[0].text.strip()


# Set up the API call to ElevenLabs for TTS
data = {
  "text": script,
  "model_id": "eleven_monolingual_v1",
  "voice_settings": {
    "stability": 0.43,
    "similarity_boost": 0.28,
    "style": 0.23,
    "use_speaker_boost": 0
  }
}


# Fetch and store the main audio in memory
response = requests.post(url, json=data, headers=headers)
main_audio_buffer = BytesIO()

for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
    if chunk:
        main_audio_buffer.write(chunk)

# Reset buffer position to the start
main_audio_buffer.seek(0)

# Load the main audio for processing
sound1 = AudioSegment.from_file(main_audio_buffer, format="mp3")

# Now get the background music from the Azure Storage Account
# Azure setup for background audio
connect_str = 'DefaultEndpointsProtocol=https;AccountName=backgroundaudio;AccountKey=JghVljN/kQzr8z+HgSLlpGP8On2JZF94Yaxxh1maDMoTtjBEyIz3Q0q9lZEi8nQ4D6LMHZg5Icru+AStoo2Zdg==;EndpointSuffix=core.windows.net'  # Replace with your Azure connection string
container_name = 'backgroundaudiofiles'  # Replace with your container name
blob_name = 'RelaxBackgroundAudio.mp3'  # Blob name for the background audio

blob_service_client = BlobServiceClient.from_connection_string(connect_str)
blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

# Fetch and store the background audio in memory
background_audio_buffer = BytesIO()
background_audio_stream = blob_client.download_blob()

background_audio_stream.readinto(background_audio_buffer)
background_audio_buffer.seek(0)  # Reset buffer position to the start

# Load the background audio for processing
background = AudioSegment.from_file(background_audio_buffer, format="mp3")
background = background - 20  # Reduces volume by 20 dB

# Load the background audio from a file
#background = AudioSegment.from_file("RelaxBackgroundAudio.mp3")
#background = background - 20  # Reduces volume by 20 dB

# Overlay background on sound1
combined = sound1.overlay(background)

# Export 'combined' to a file or use it as needed
combined.export(f"{client_name}_with_BG_audio.mp3", format='mp3')







