from openai import OpenAI


def text_to_speech(input_text, output_file="output.mp3", model="tts-1", voice="alloy"):
    client = OpenAI()

    response = client.audio.speech.create(
        model=model,
        voice=voice,
        input=input_text
    )

    response.stream_to_file(output_file)

if __name__ == "__main__":
    input_text = "Hello world! This is a streaming test."
    text_to_speech(input_text)
