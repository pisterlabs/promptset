import os
from pydub import AudioSegment
from concurrent.futures import ThreadPoolExecutor
import whisper
import nltk
from nltk.tokenize import sent_tokenize
from Open_AI_functions import OpenAIClient

# Constants
ONE_HOUR_MS = 60 * 60 * 1000  # One hour in milliseconds
SENTENCES_PER_SEGMENT = 4  # Number of sentences per text segment for embedding

# Ensure necessary NLTK data is downloaded
nltk.data.path.append('/tmp/nltk_data')
nltk.download('punkt', download_dir='/tmp/nltk_data')

# Function to process each audio segment
def process_segment(model, segment, index, base_filepath):
    segment_file = f"{base_filepath}_part{index}.wav"
    segment.export(segment_file, format="wav")
    result = model.transcribe(segment_file)
    os.remove(segment_file)  # Remove the segment file after transcription
    return result["text"]

def split_audio(filepath, num_workers=4):
    """
    Splits an audio file into one-hour segments and transcribes them using parallel processing.
    :param filepath: Path to the audio file.
    :param num_workers: Number of parallel workers for processing.
    :return: Full transcribed text of the audio.
    """
    try:
        audio = AudioSegment.from_file(filepath)
        duration = len(audio)
        full_transcript = ""
        model = whisper.load_model("base")
        base_filepath = os.path.splitext(filepath)[0]

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for start in range(0, duration, ONE_HOUR_MS):
                end = min(start + ONE_HOUR_MS, duration)
                segment = audio[start:end]
                futures.append(executor.submit(process_segment, model, segment, start // ONE_HOUR_MS, base_filepath))

            for future in futures:
                full_transcript += future.result() + " "

        print('full_transcript', full_transcript)
        return full_transcript.strip()
    except Exception as e:
        print(f"Error processing audio file: {e}")
        return None

def transcribe_audio(full_transcript):
    """
    Generates embeddings for segments of the transcript.
    :param full_transcript: The full text transcript of an audio file.
    :return: Dictionary of embeddings for each text segment.
    """
    embeddings = {}
    index = 1
    sentences = sent_tokenize(full_transcript)

    openai_client = OpenAIClient()

    for i in range(0, len(sentences), SENTENCES_PER_SEGMENT):
        segment = " ".join(sentences[i : i + SENTENCES_PER_SEGMENT])
        embedding = openai_client.get_embeddings(segment)
        embeddings[index] = [embedding, segment]
        index += 1

    return embeddings
