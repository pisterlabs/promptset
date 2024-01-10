"""
Storyteller: A simple audio storytelling app using OpenAI API.

Example Usage:
    python storyteller.py --address=127.0.0.1 --port=7860
"""
import argparse
import base64
import config
import gradio as gr
import io
import json
import openai
import os
import requests
import subprocess

from config import SpeechMethod
from google.cloud import texttospeech
from typing import Dict, List, Tuple


# Set OpenAI API Key
openai.api_key = os.environ.get("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("OpenAI API Key not set as environnment variable OPENAI_API_KEY")

# Get eleven.io
elevenio_api_key = None
if config.SPEECH_METHOD == SpeechMethod.ELEVENIO:
    elevenio_api_key = os.environ.get("ELEVENIO_API_KEY")
    if elevenio_api_key is None:
        raise ValueError(
            "Eleven.io API Key not set as environnment variable ELEVENIO_API_KEY"
        )

# Initial message
messages = [
    {
        "role": "system",
        "content": config.INITIAL_PROMPT,
    }
]


"""
Main functions
"""


def transcribe_audio(audio_file: str) -> str:
    """
    Transcribe audio file using OpenAI API.

    Args:
        audio: stringified path to audio file. WAV file type.

    Returns:
        str: Transcription of audio file
    """
    # gradio sends in a .wav file type, but it may not be named that. Rename with
    # .wav extension because Whisper model only accepts certain file extensions.
    if not audio_file.endswith(".wav"):
        os.rename(audio_file, audio_file + ".wav")
        audio_file = audio_file + ".wav"

    # Open audio file and transcribe
    with open(audio_file, "rb") as f:
        transcript = openai.Audio.transcribe("whisper-1", f)
    text_transcription = transcript["text"]

    return text_transcription


def chat_complete(
    text_input: str, messages: List[Dict[str, str]]
) -> Tuple[str, List[Dict[str, str]]]:
    """
    Chat complete using OpenAI API. This is what generates stories.

    Args:
        text_input: Text to use as prompt for story generation
        messages: List of previous messages

    Returns:
        str: Generated story
        messages: Updated list of messages
    """
    # Init with prompt on first call
    if not messages:
        messages = [
            {
                "role": "system",
                "content": config.INITIAL_PROMPT,
            }
        ]

    # Append to messages for chat completion
    messages.append({"role": "user", "content": text_input})

    # Fetch response from OpenAI
    print("Messages sent to call: ", messages)
    response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)

    # Extract and store message
    system_message = dict(response["choices"][0]["message"])
    messages.append(system_message)

    # Return message to display
    display_message = system_message["content"]

    if config.SPEECH_METHOD == SpeechMethod.MAC:
        # call subprocess in background
        subprocess.Popen(["say", system_message["content"]])

    return display_message, messages


def generate_image(text_input: str) -> str:
    """
    Generate an image using DALL-E via OpenAI API.

    Args:
        text_input: Text to use as prompt for image generation

    Returns:
        str: Path to generated image
    """
    prompt = text_input[: config.PROMPT_MAX_LEN]
    response = openai.Image.create(prompt=prompt, n=1, size=config.RESOLUTION)
    image_url = response["data"][0]["url"]
    img_data = requests.get(image_url).content
    with open(config.IMAGE_PATH, "wb") as handler:
        handler.write(img_data)
    return config.IMAGE_PATH


def audio_file_to_html(audio_file: str) -> str:
    """
    Convert audio file to HTML audio player.

    Args:
        audio_file: Path to audio file

    Returns:
        audio_player: HTML audio player that auto-plays
    """
    # Read in audio file to audio_bytes
    audio_bytes = io.BytesIO()
    with open(audio_file, "rb") as f:
        audio_bytes.write(f.read())

    # Generate audio player HTML object for autoplay
    audio_bytes.seek(0)
    audio = base64.b64encode(audio_bytes.read()).decode("utf-8")
    audio_player = (
        f'<audio src="data:audio/mpeg;base64,{audio}" controls autoplay></audio>'
    )
    return audio_player


def text_to_speech_gcp(input_text: str, tts_voice_label: str) -> str:
    """
    Use GCP Text-to-Speech API to convert text to a WAV file.

    Args:
        input_text: Text to convert to speech
        tts_voice_label: Label of voice to use, from keys of TTS_VOICE_OPTIONS in config

    Returns
        str: Path to output audio file
    """
    print(f"Convert text to speech: {input_text}")
    # set up the client object
    client = texttospeech.TextToSpeechClient()

    # set up the synthesis input object
    synthesis_input = texttospeech.SynthesisInput(text=input_text)

    # derive language code and ID
    tts_voice_id = config.TTS_VOICE_OPTIONS[tts_voice_label]
    tts_language_code = "-".join(tts_voice_id.split("-")[0:2])

    # set up the voice parameters
    voice = texttospeech.VoiceSelectionParams(
        language_code=tts_language_code, name=tts_voice_id
    )

    # set up the audio parameters
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
    )

    # generate the request
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # save the response audio as an MP3 file
    with open(config.GENERATED_SPEECH_PATH, "wb") as out:
        out.write(response.audio_content)

    # Generate audio player HTML object for autoplay
    audio_player = audio_file_to_html(config.GENERATED_SPEECH_PATH)

    return audio_player


def text_to_speech_elevenio(
    input_text: str,
    tts_voice_id: str,
    stability: float = 0.65,
    similarity_boost: float = 0.85,
) -> str:
    """
    Use Eleven.io Text-to-Speech API to convert text to a WAV file.

    Args:
        input_text: Text to convert to speech
        tts_voice_label: Label of voice to use, from keys of ELEVENIO_VOICE_ID in config
        similarity_boost: Similarity boost for voice
        stability: Stability for voice

    Returns
        str: Path to output audio file
    """
    print(f"Convert text to speech: {input_text}")
    tts_voice_id = config.ELEVENIO_VOICE_ID  # Use pre-assigned from config
    url = f"{config.ELEVENIO_TTS_BASE_URL}/{tts_voice_id}"

    payload = json.dumps(
        {
            "text": input_text,
            "voice_settings": {
                "stability": stability,
                "similarity_boost": similarity_boost,
            },
        }
    )
    headers = {
        "xi-api-key": elevenio_api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    # save the response audio as an MP3 file
    with open(config.GENERATED_SPEECH_PATH, "wb") as out:
        out.write(response.content)

    # Generate audio player HTML object for autoplay
    audio_player = audio_file_to_html(config.GENERATED_SPEECH_PATH)

    # return response.audio_content
    return audio_player


"""
Gradio UI Definition
"""
with gr.Blocks(analytics_enabled=False, title="VocalTales: Audio Storyteller") as ui:
    # Session state box containing all user/system messages, hidden
    messages = gr.State(list())

    # Initialize TTS
    tts_fn = None
    if config.SPEECH_METHOD == SpeechMethod.GCP:
        tts_fn = text_to_speech_gcp
    elif config.SPEECH_METHOD == SpeechMethod.ELEVENIO:
        tts_fn = text_to_speech_elevenio

    # Set up layout and link actions together
    with gr.Row():
        with gr.Column(scale=1):
            with gr.Accordion("Click for Instructions & Configuration:", open=False):
                # Voice Selection Dropdown
                voice_labels = [k for k in config.TTS_VOICE_OPTIONS.keys()]
                voice_selection = gr.Dropdown(
                    choices=voice_labels,
                    value=config.TTS_VOICE_DEFAULT,
                    label="Voice Selection",
                )

                # Instructions
                gr.Markdown(config.INSTRUCTIONS_TEXT)

            # Audio Input Box
            audio_input = gr.Audio(
                source="microphone", type="filepath", label="User Audio Input"
            )

            # User Input Box
            transcribed_input = gr.Textbox(label="Transcription")

            # Story Output Box
            story_msg = gr.Textbox(label="Story")

            if tts_fn:
                # Connect story output to audio output after calling TTS on it
                html = gr.HTML()
                story_msg.change(tts_fn, [story_msg, voice_selection], html)

        with gr.Column(scale=1):
            # Story Generated Image
            gen_image = gr.Image(label="Story Image", shape=(None, 5))

    # Connect audio input to user input
    audio_input.change(transcribe_audio, audio_input, transcribed_input)

    # Connect user input to story output
    transcribed_input.change(
        chat_complete, [transcribed_input, messages], [story_msg, messages]
    )

    # Connect story output to image generation
    story_msg.change(generate_image, story_msg, gen_image)


if __name__ == "__main__":
    # Add a address string argument that defaults to 127.0.0.1
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--address",
        type=str,
        default="127.0.0.1",
        help="""
        Address to run the server on. 127.0.0.1 for local. 0.0.0.0 for "
        remote or docker
        """,
    )
    # add a port with None default
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Port to run the server on",
    )
    parser.add_argument(
        "--username",
        type=str,
        default=None,
        help="Username for basic auth",
    )
    parser.add_argument(
        "--password",
        type=str,
        default=None,
        help="Password for basic auth",
    )
    args = parser.parse_args()

    # Configure auth
    if args.username and args.password:
        auth = (args.username, args.password)
    else:
        auth = None

    # Launch UI
    ui.launch(server_name=args.address, server_port=args.port, auth=auth)
