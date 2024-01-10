import requests
from pydub import AudioSegment
from pydub.silence import detect_silence
from dotenv import load_dotenv
import os
import subprocess
import logging
import openai
from google.cloud import speech
import tkinter as tk
from tkinter import filedialog

# Configure logging
logging.basicConfig(filename='app.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')

# Load environment variables
load_dotenv()

# API key for OpenAI
api_key = os.getenv('OPENAI_API_KEY')

# Function to choose transcription service
def choose_transcription_service():
    print("Choose the transcription service:")
    print("1: Whisper")
    print("2: Google Cloud Speech-to-Text")
    choice = input("Enter your choice (1 or 2): ")
    return choice

# Function to summarize text using OpenAI
def summarize_text(text, model="text-davinci-003", max_tokens=1000):
    # Updated prompt for a detailed and comprehensive summary
    prompt = (
        "Please provide a detailed and comprehensive summary, aiming for about 15% of the original text's length. "
        "Include the main points, action items, references, stories, sentiment, follow-up questions, arguments, and related topics.\n\n"
        f"Text: {text}\n"
    )
    response = openai.completions.create(
        model=model,
        prompt=prompt,
        max_tokens=max_tokens,
    )
    return response.choices[0].text.strip()

# Function to break the text into smaller segments and summarize each
def split_text_into_segments(text, max_length=3000):
    words = text.split()
    segments = []
    current_segment = []

    for word in words:
        if len(' '.join(current_segment + [word])) > max_length:
            segments.append(' '.join(current_segment))
            current_segment = [word]
        else:
            current_segment.append(word)
    
    if current_segment:
        segments.append(' '.join(current_segment))
    
    return segments

# Function to split text into segments
def split_text_into_segments(text, max_length=3000):
    words = text.split()
    segments = []
    current_segment = []

    for word in words:
        if len(' '.join(current_segment + [word])) > max_length:
            segments.append(' '.join(current_segment))
            current_segment = [word]
        else:
            current_segment.append(word)
    
    if current_segment:
        segments.append(' '.join(current_segment))
    
    return segments

# Function to transcribe with Whisper
def transcribe_with_whisper(segment_path):
    try:
        with open(segment_path, 'rb') as segment_file:
            response = requests.post(
                'https://api.openai.com/v1/audio/transcriptions',
                headers={
                    'Authorization': f'Bearer {api_key}',
                    'OpenAI-Model': 'whisper-1'
                },
                files={'file': (os.path.basename(segment_path), segment_file)},
            )
            if response.status_code == 200:
                return response.json().get('text', '')
            else:
                logging.error(f"Whisper transcription error. Status code: {response.status_code}")
    except Exception as e:
        logging.error(f"Whisper transcription exception: {e}")

# Function to transcribe with Google Cloud
def transcribe_with_google_cloud(segment_path, sample_rate):
    try:
        client = speech.SpeechClient()
        print(f"Transcribing with Google Cloud: {segment_path}")

        with open(segment_path, 'rb') as audio_file:
            content = audio_file.read()

        if not content:
            print(f"Warning: Empty audio content for {segment_path}")
            return ""

        audio = speech.RecognitionAudio(content=content)
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=sample_rate,  # Use the provided sample rate
            language_code="en-US",
            enable_automatic_punctuation=True
        )

        response = client.recognize(config=config, audio=audio)
        transcription = ' '.join([result.alternatives[0].transcript for result in response.results])
        print(f"Google Cloud Transcription: {transcription[:50]}...")
        return transcription

    except Exception as e:
        print(f"Error during Google Cloud transcription: {e}")
        return ""

# Main script starts here
root = tk.Tk()
root.withdraw()

input_audio_file_path = filedialog.askopenfilename(
    title="Select Audio File", 
    filetypes=[("Audio Files", "*.m4a *.wav *.mp3 *.ogg *.flac")]
)
print(f"Selected audio file: {input_audio_file_path}")
if not input_audio_file_path:
    logging.error("No file selected.")
    exit(1)
print(f"Converting audio file to WAV format...")

output_folder = "output_segments"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

input_filename, input_extension = os.path.splitext(os.path.basename(input_audio_file_path))
output_audio_file_path = os.path.join(os.path.dirname(input_audio_file_path), f'{input_filename}.wav')

if input_extension.lower() not in ['.wav', '.mp3', '.ogg', '.flac']:
    ffmpeg_executable = 'C:\\ffmpeg\\bin\\ffmpeg.exe'
    ffmpeg_command = [ffmpeg_executable, '-i', input_audio_file_path, output_audio_file_path]
    try:
        subprocess.run(ffmpeg_command, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"FFmpeg conversion error: {e}")
        exit(1)
else:
    output_audio_file_path = input_audio_file_path
print("Conversion completed.")

# Audio Processing
print("Creating audio segments...")

# Load the audio file
audio = AudioSegment.from_file(output_audio_file_path)
mono_audio = audio.set_channels(1)  # Convert to mono

# Parameters for silence detection
min_silence_len = 1000  # Silence length to consider it as a split point (in ms)
silence_thresh = -40   # Silence threshold (in dBFS)
min_segment_len = 30 * 1000  # Minimum segment length (30 seconds)

# Initialize variables
start = 0
segment_paths = []
sample_rates = []  # List to store sample rates of segments

while start < len(mono_audio):
    end = min(start + min_segment_len, len(mono_audio))
    segment = mono_audio[start:end]
    if len(segment) > 0:
        segment_path = os.path.join(output_folder, f'segment_{len(segment_paths)}.wav')
        segment.export(segment_path, format="wav")  # This line should now work without error
        segment_paths.append(segment_path)
        sample_rates.append(segment.frame_rate)  # Store the sample rate of the segment
    start = end

print(f"Created {len(segment_paths)} segments for transcription.")

service_choice = choose_transcription_service()
print(f"Chosen transcription service: {service_choice}")
transcriptions = []

for segment_path, rate in zip(segment_paths, sample_rates):
    if service_choice == '1':
        transcription = transcribe_with_whisper(segment_path)
    elif service_choice == '2':
        transcription = transcribe_with_google_cloud(segment_path, rate)
    else:
        logging.warning("Invalid transcription service choice, defaulting to Whisper.")
        transcription = transcribe_with_whisper(segment_path)
    if transcription:
        transcriptions.append(transcription)
    print(f"Transcribed segment: {segment_path}")
print(f"Completed transcription of all segments.")

# Combine all transcriptions into one string
combined_transcription = ' '.join(transcriptions)

# Split the large text into smaller segments
segments = split_text_into_segments(combined_transcription, max_length=3000)

# Summarize each segment and combine them
all_summaries = []
for segment in segments:
    summary = summarize_text(segment)
    all_summaries.append(summary)
combined_summary = ' '.join(all_summaries)

transcriptions_output_path = os.path.join(os.path.dirname(input_audio_file_path), f"{input_filename}_Transcription.txt")
with open(transcriptions_output_path, 'w', encoding='utf-8') as file:
    file.write(combined_transcription)

logging.info(f"Transcription saved to {transcriptions_output_path}")

print("Starting summarization process...")
print("Summarization completed.")

# Construct the combined_content without using f-string
print("Writing output to file...")
combined_content = "Transcriptions:\n\n" + combined_transcription + "\n\nSummaries:\n" + combined_summary

final_combined_output_path = os.path.join(os.path.dirname(input_audio_file_path), f"Combined_{input_filename}_Summary.txt")
with open(final_combined_output_path, 'w', encoding='utf-8') as combined_file:
    combined_file.write(combined_content)

logging.info(f"Combined transcriptions and summaries saved to {final_combined_output_path}")
print(f"Output written to {final_combined_output_path}")