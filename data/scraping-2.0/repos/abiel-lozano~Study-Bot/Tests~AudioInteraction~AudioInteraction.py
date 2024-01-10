import pyaudio
import wave
import ffmpeg
import whisper
import openai
from pathlib import Path
from elevenlabs import generate, play, set_api_key

openai.api_key = '' # OpenAI API key
set_api_key('') # Elevenlabs API key
GPT_MODEL = "gpt-3.5-turbo"

def record():
    # Recording parameters:
	CHUNK = 1024 # Chunk size
	FORMAT = pyaudio.paInt16 # Audio codec format
	CHANNELS = 2
	RATE = 44100 # Sample rate
	RECORD_SECONDS = 5 # Recording duration
	WAVE_OUTPUT_FILENAME = 'output.wav'

	audio = pyaudio.PyAudio()

	# Open audio stream for recording
	stream = audio.open(format = FORMAT, channels = CHANNELS, rate = RATE, input = True, frames_per_buffer = CHUNK)
	print('Recording question...')

	# Initalize audio buffer
	frames = []

	# Record audio stream in chunks
	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
		data = stream.read(CHUNK)
		frames.append(data)

	print('Recording stopped.')

	# Stop and close audio stream
	stream.stop_stream()
	stream.close()
	audio.terminate()

	# Save the recorded audio as a WAV file
	wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	wf.setnchannels(CHANNELS)
	wf.setsampwidth(audio.get_sample_size(FORMAT))
	wf.setframerate(RATE)
	wf.writeframes(b''.join(frames))
	wf.close()
	
	# Convert WAV to MP3
	input_audio = ffmpeg.input(WAVE_OUTPUT_FILENAME)
	output_audio = ffmpeg.output(input_audio, 'output.mp3')
	ffmpeg.run(output_audio)
	
	print('File saved as "output.mp3"')

record()

model = whisper.load_model('base')
result = model.transcribe('output.wav', fp16=False, language='English')
print(result['text'])

# If there is a file with the name "output.mp3" in the directory, delete it.
if Path('output.mp3').is_file():
	Path('output.mp3').unlink()
# If there is a file with the name "output.wav" in the directory, delete it.
if Path('output.wav').is_file():
	Path('output.wav').unlink()

# Add infromation source
source = """

"""

# Build the prompt
query = f"""Try and use the information below to answer the question. If the 
question is unrelated to the information, ignore the information, and try to
answer the question without it.
Information:
\"\"\"
{source}
\"\"\"
Question: {result}"""

response = openai.ChatCompletion.create(
	messages = [
		{'role': 'system', 'content': 'You answer questions in the same language as the question.'},
        {'role': 'user', 'content': query},
	],
	model = GPT_MODEL, temperature = 0
)	

answer = response['choices'][0]['message']['content']
print('Answer: ', answer)

audioOutput = generate(answer)
play(audioOutput)