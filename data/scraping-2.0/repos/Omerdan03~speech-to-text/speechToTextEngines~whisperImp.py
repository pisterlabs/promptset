import whisper
from openai import OpenAI
import timeit

N = 10


def transcribe(audio_path, model_name="base.en"):
	model = whisper.load_model(model_name)
	result = model.transcribe(audio_path)
	return result["text"]


def transcribe_timeit(audio_path, model_name="base.en"):
	model = whisper.load_model(model_name)

	def transcribe_code():
		_ = model.transcribe(audio_path)

	execution_time = timeit.timeit(stmt=transcribe_code, number=N) / N
	return execution_time


def transcribe_api(audio_path):
	client = OpenAI()

	audio_file = open(audio_path, "rb")

	result = client.audio.transcriptions.create(
		model="whisper-1",
		file=audio_file
	)
	return result.text


def transcribe_api_timeit(audio_path):
	client = OpenAI()
	audio_file = open(audio_path, "rb")

	def transcribe_code():
		_ = client.audio.transcriptions.create(
			model="whisper-1",
			file=audio_file)

	execution_time = timeit.timeit(stmt=transcribe_code, globals=globals(), number=N) / N
	return execution_time
