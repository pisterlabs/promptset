# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai

from .utils import save_audio_file
from ..gpt_generate import generate_corrected_transcript, generate_corrected_text
from ..utils import get_apikey

# openai.api_key = "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"
# audio_file= open("record/Bridletowne Cir 2.m4a", "rb")
# transcript = openai.Audio.translate("whisper-1", audio_file)
# print(transcript)
get_apikey(openai)


def gpt_audio_response(audio_file, user, combined_request):
    audio_path = save_audio_file(audio_file, user)
    audio_file_new = open(audio_path, "rb")
    generate = generate_corrected_transcript(0.8, audio_file_new, combined_request)
    return generate


def gpt_text_response(text_file, combined_request):
    return generate_corrected_text(0.8, text_file, combined_request)
