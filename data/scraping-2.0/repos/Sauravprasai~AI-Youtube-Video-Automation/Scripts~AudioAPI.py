from openai import OpenAI

def generate_audio():
    with open('Text Folder/API_key.txt', 'r') as file:
        api_key = file.read().strip()

    with open('Text Folder/Response.txt', 'r') as file:
        prompt = file.read()

    client = OpenAI(
        api_key=api_key,
    )

    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=f"{prompt}",
    )

    response.stream_to_file("Audio/output.mp3")
    print("Audio file generated successfully!")