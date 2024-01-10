import os
import queue
import openai
import sounddevice as sd
import soundfile as sf
from pydub.playback import play
from langdetect import detect
from termcolor import colored
import threading

# Read the OpenAI API key from an environment variable
API_KEY = os.environ.get('OPENAI_API_KEY')

# Set the number of seconds to record and send to OpenAI
RECORD_SECONDS = 5

# Set the sample rate of the audio data
SAMPLE_RATE = 32000

# Define the OpenAI Completion API parameters
model_engine = "text-davinci-003"
temperature = 0.7
max_tokens = 60

# Define the text-to-speech parameters
tts_language = 'en'
tts_slow = False

# Create an instance of the OpenAI API client
openai.api_key = API_KEY

# Define the prompt template for our main purpose, i.e. telling jokes.
prompt_template = """You are a dad-joke assistant. Reply with a funny dad-joke related to the transcription below:
{summary}"""

# Define the prompt template for summarization
summarization_template = """Summarize the following transcription of a conversation:
{transcript}
"""

# Circular buffer of last 30 / RECORD_SECONDS user utterances
transcript_queue = queue.Queue(maxsize=30 // RECORD_SECONDS)

def transcript_queue_processor():

    # Keep a buffer of last 10 transcriptions that we constantly summarize for continuity
    transcript_buffer = []

    while True:

        # Wait for the next transcript to be added to the queue
        transcript_latest = transcript_queue.get()

        # Add the latest transcript to the transcript buffer
        transcript_buffer.append(transcript_latest)

        # If the buffer is full, remove the oldest transcript
        if len(transcript_buffer) > 10:
            transcript_buffer.pop(0)

        # Print transcript buffer
        #print("Transcript buffer: " + str(transcript_buffer))

        # Replace the {transcript} placeholder in the summarization template with a newline joined transcript_buffer
        summarization_prompt = summarization_template.format(transcript='\n'.join(transcript_buffer))

        # Send the prompt to the OpenAI Summarization API
        summarization_response = openai.Completion.create(
            engine=model_engine,
            prompt=summarization_prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )

        # Save the summary
        summary = summarization_response.choices[0].text.strip()

        # Print the summary
        print("Summary: " + summary)

        # Replace the {summary} placeholder in the prompt template
        prompt = prompt_template.format(summary=summary)

        # Send the prompt to the OpenAI Completion API
        response = openai.Completion.create(
            engine=model_engine,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens
        )
        response_text = response.choices[0].text.strip()

        # Print the response from the OpenAI Completion API
        colored_response_text = colored(response_text, 'green')
        print("Suggested joke: " + colored_response_text)

# Start the thread to process the transcript queue
threading.Thread(target=transcript_queue_processor).start()

print("Listening... (say bye to stop)")

# Loop forever, recording and sending    audio data every RECORD_SECONDS
while True:
  
    # Record audio data from the microphone
    audio_data = sd.rec(int(SAMPLE_RATE * RECORD_SECONDS), samplerate=SAMPLE_RATE, channels=1, dtype='int16')

    # Wait for the recording to complete
    sd.wait()

    # Save the audio data to a file
    file_name = 'temp.wav'
    sf.write(file_name, audio_data, SAMPLE_RATE)

    # Transcribe the audio file using the OpenAI API
    with open(file_name, "rb") as audio_file:
        transcript = openai.Audio.transcribe("whisper-1", audio_file)

    # Save the transcribed text into input.
    input = transcript.text.strip()

    # If the input string is empty or blank, skip the rest of the loop
    if not input:
        continue

    # Print the transcribed text
    #print("Human: " + input)

    # Keep a circular buffer of last 30 / RECORD_SECONDS user utterances
    transcript_queue.put(input, block=False)

    # If the user said "bye", regardless of case, stop the program
    if input.lower() == "bye.":
        break

# Wait for queue to finish processing
transcript_queue.join()