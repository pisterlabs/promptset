"""
This module replicates the stuttering issue with weleven labs stream mode.
"""
import os

import openai
import elevenlabs
import dotenv


def load():
    """take environment variables from .env."""
    dotenv.load_dotenv()

    # Create an OpenAI client
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Set the Eleven API key
    elevenlabs.set_api_key(os.getenv("ELEVENLABS_API_KEY"))

    # Load the path where the mpv player is located.
    # This is used to play the audio with elevenlabs.stream()
    os.environ["PATH"] += os.pathsep + os.getenv("MPV_PATH")

    return client

def text_stream(prompt: str, client: openai.OpenAI, print_output: bool = True):
    """Yield text from the OpenAI stream as it arrives"""

    # Prompt for the user to say something
    openai_stream = client.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )

    for chunk in openai_stream:
        if chunk.choices[0].delta.content is not None:
            if print_output:
                print(chunk.choices[0].delta.content, end="")
            yield chunk.choices[0].delta.content


if __name__ == "__main__":
    # Get OpenAi client
    client = load()

    # Create a voice, if you have issues with the voice_id, check the documentation or use this line instead:
    # voice = "Nicole"
    voice = elevenlabs.Voice(voice_id="chQ8GR2cY20KeFjeSaXI")

    user_prompt = "Summarize the \"Little Red Riding Hood\" story in Spanish."

    # Stream audio in real-time, as it's being generated.
    audio_stream = elevenlabs.generate(
        text=text_stream(user_prompt, client),
        voice=voice,
        model="eleven_multilingual_v2",
        stream=True
    )

    # Play the audio stream
    elevenlabs.stream(audio_stream)
