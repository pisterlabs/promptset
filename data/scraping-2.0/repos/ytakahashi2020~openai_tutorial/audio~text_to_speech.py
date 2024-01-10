from openai import OpenAI

client = OpenAI()

response = client.audio.speech.create(
    model="tts-1-hd",
    voice="nova",
    input="Hallo, My Name is Yuki.",
)

response.stream_to_file("output.mp3")
