from ninja.files import UploadedFile
import replicate
import openai
from core.models import SpeechModel
from speech2text.settings import env
import requests
import uuid
from django.core.files import File
from django.core.files.temp import NamedTemporaryFile


def speech2text_serivce(us_file: str, uploaded_file: UploadedFile):
    """
    Speech to text service. TODO: get rid of saved file
    """
    output = replicate.run(
        "openai/whisper:e39e354773466b955265e969568deb7da217804d8e771ea8c9cd0cef6591f8bc",
        input={"audio": open(us_file, "rb")},
    )
    SpeechModel.objects.create(
        transcription=output["transcription"],
        segments=output["segments"],
        audio_file=uploaded_file,
    )
    return {"text": output}


def chatgpt_service(text: str):
    """
    Call chatgpt service. TODO: user input
    """
    openai.api_key = env("OPENAI_API_KEY")
    out = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a funny person."},
            {"role": "user", "content": text},
            # {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            # {"role": "user", "content": "Where was it played?"}
        ],
    )
    return out
    # User: User message
    # System: System message which provides context for the assistant: E.G insert system message
    # Assistant: AI assistant who answers the prompt

    # TODO: Probably abandon openai library? Depends on how to use llamaindex
    # API reference: https://platform.openai.com/docs/api-reference/introduction


def elevenlabs_service(text: str):
    # voice is rachel
    CHUNK_SIZE = 1024
    voice_id = "21m00Tcm4TlvDq8ikWAM"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": str(env("ELEVENLABS_API_KEY")),
    }
    data = {
        "text": text,
        "model_id": "eleven_monolingual_v1",
        "voice_settings": {"stability": 0, "similarity_boost": 0},
    }
    # Post response
    response = requests.post(url, json=data, headers=headers)
    # Create temporary file
    temporary_filename = str(uuid.uuid4()) + ".mp3"
    temporary_audio_file = NamedTemporaryFile(delete=True)
    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
        if chunk:
            temporary_audio_file.write(chunk)
    # Save temporary file and convert to django file object
    temporary_audio_file.flush()
    temp_file = File(temporary_audio_file, name=temporary_filename)
    SpeechModel.objects.create(transcription=text, audio_file=temp_file)
    return temporary_filename


# TODO: Speech to speech service with chatgpt

# https://gpt-index.readthedocs.io/en/latest/how_to/customization/custom_llms.html
# https://gpt-index.readthedocs.io/en/latest/guides/tutorials/building_a_chatbot.html
