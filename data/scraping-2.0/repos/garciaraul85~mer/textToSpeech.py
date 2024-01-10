import openai
client = openai.OpenAI()

speech_file_path = "speech.mp3"

response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input="Hello world"
)

response.stream_to_file(speech_file_path)