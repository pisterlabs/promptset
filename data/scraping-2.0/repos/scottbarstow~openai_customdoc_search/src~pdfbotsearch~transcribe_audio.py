import os
from openai_utils import get_openai_api_key
import openai


openai_api_key = get_openai_api_key()  
openai.api_key = openai_api_key

file_path = os.path.expanduser('~/Desktop/Auronix.mp4')

file = open(file_path, "rb")

transcript = openai.Audio.translate("whisper-1", file)

print(transcript)
