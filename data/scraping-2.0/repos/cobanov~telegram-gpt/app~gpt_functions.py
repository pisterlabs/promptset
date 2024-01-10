from openai import OpenAI
from dotenv import load_dotenv
from elevenlabs import generate, set_api_key
import os
import sys

load_dotenv()

try:
    client = OpenAI()
    set_api_key(os.environ["ELEVENLABS_API_KEY"])
except KeyError:
    print("Environment variables not set properly.")
    sys.exit(1)

# Mapping actors to their specific settings
actor_settings = {
    "David": {
        "voice_id": "ta7FSUbII3HcuToiCxEM",
        "system_message": "You are Sir David Attenborough. Be snarky and funny. Don't repeat yourself. Make it short.",
    },
    "Jane": {
        "voice_id": "XB0fDUnXU5powFXDhCwa",
        "system_message": "You are Jane Austin. Be snarky and funny. Don't repeat yourself. Make it short.",
    },
}


def get_gpt_response(message, actor):
    if actor not in actor_settings:
        raise ValueError(f"Actor {actor} not recognized.")

    print(f"GPT response as {actor}")

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": actor_settings[actor]["system_message"]},
                {"role": "user", "content": message},
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in generating GPT response: {e}")
        return None


def get_audio_file(text, actor):
    if actor not in actor_settings:
        raise ValueError(f"Actor {actor} not recognized.")

    print(f"ElevenLabs response as {actor}")

    try:
        audio = generate(text, voice=actor_settings[actor]["voice_id"])
        return audio
    except Exception as e:
        print(f"Error in generating audio file: {e}")
        return None


if __name__ == "__main__":
    response = get_gpt_response("Hello, I am Mert. Who are you?", "David")
    if response:
        print(response)
