from openai import OpenAI

from .gpt_model_enum import GptModelDefines
from .gpt_utils import get_model_name

client = OpenAI()

response = client.audio.speech.create(
	model=get_model_name(GptModelDefines.TTS),
	voice="shimmer",
	input="Hello world! This is a streaming test.",
)

print(response)
response.stream_to_file("output.aac")
