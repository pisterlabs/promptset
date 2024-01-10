from robocorp.tasks import task
from robocorp import workitems, vault
import requests
import openai

@task
def transcribe_hf_inference():
    '''Transcribe audio files using Whisper from the organization's Huggingface Inference Endpoints.'''

    hf_secret = vault.get_secret("Huggingface")
    headers = {
        "Authorization": f"Bearer {hf_secret['api-token']}",
        "Content-Type": "audio/flac",
    }

    for item in workitems.inputs:
        paths = item.download_files("*.flac")

        for path in paths:
            file = open(path, "rb")
            response = requests.post(hf_secret['whisper-url'], headers=headers, data=file)
            json_resp = response.json()
            print(f"HF/AI SAYS: {json_resp['text']}")
            workitems.outputs.create(payload=json_resp)

@task
def transcribe_openai():
    '''Transcribe audio files using Whisper from OpenAI API.'''

    oa_secret = vault.get_secret("OpenAI")
    openai.api_key = oa_secret["key"]

    for item in workitems.inputs:
        paths = item.download_files("*.m4a")

        for path in paths:
            file = open(path, "rb")
            transcript = openai.Audio.transcribe("whisper-1", file)
            print(f"OPENAI SAYS: {transcript['text']}")
            workitems.outputs.create(payload=transcript)
