import os
from pathlib import Path
from openai import OpenAI
import subprocess
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

openai = OpenAI(api_key="")

def read_book(file_path):
    logging.info(f"Reading book from {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

messages = [
    {
        "role": "system",
        "content": """
        Help me guess speakers for different quotes in a paragraph. 
        Guess the gender of the speaker if you can. If unknown, say UNKNOWN.
        Give output for each quote in the format: speaker,gender
        """
    }
]

def identify_speaker(context, quote):
    global messages
    # Add the new user prompt as a message
    user_message = {
        "role": "user",
        "content": f"Given the context: [{context}], who is most likely to say: [{quote}]?"
    }
    messages.append(user_message)
    
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=60
    )
    print(response)

    # Update messages with the response
    assistant_message = response.choices[0].message
    response_message = {
        "role": "assistant",
        "content": assistant_message.content  # Accessing content directly
    }
    messages.append(response_message)

    # Process response
    content = assistant_message.content
    if content is None:
        return ""
    resp = content.strip()
    try:
        speaker = resp[0].strip()
        gender = resp[1].strip() if len(resp) > 1 else "male"  # Default to male if gender is unknown
    except IndexError:
        speaker = "UNKNOWN"
        gender = "male"  # Default to male if parsing fails

    return speaker, gender


def assign_voice():
    male_voices = ["Echo", "Onyx", "Fable"]
    female_voices = ["Shimer", "Nova", "Alloy"]

    voice_map = {}
    male_index = [0]  # Use a list to maintain state
    female_index = [0]

    def voice_picker(speaker, gender: str):
        if speaker in voice_map:
            return voice_map[speaker]
        else:
            if gender.lower() == 'male':
                voice = male_voices[male_index[0] % len(male_voices)]
                male_index[0] += 1
            else:
                voice = female_voices[female_index[0] % len(female_voices)]
                female_index[0] += 1
            voice_map[speaker] = voice
            return voice
    return voice_picker

def chunk_text(input_text):
    chunks = []
    paragraphs = input_text.split('\n\n')
    logging.info(f"Split book into {len(paragraphs)} paragraphs")

    for paragraph in paragraphs:
        # Check if the paragraph is in square brackets
        if paragraph.startswith('[') and paragraph.endswith(']'):
            logging.info(f"Found system message of length: {len(paragraph)}")
            # Remove the square brackets and treat as a system message
            content = paragraph[1:-1]
            chunks.append((('system', 'male'), content))
        else:
            # Find quotes within the paragraph
            quotes = re.findall(r'"([^"]*)"', paragraph)
            logging.info(f"Found {len(quotes)} quotes in paragraph of length: {len(paragraph)}")
            start = 0

            for quote in quotes:
                end = paragraph.find(quote, start)
                # Add non-quote text as narration
                if start < end:
                    chunks.append((('narrator', 'male'), paragraph[start:end]))
                # Add quote text with identified speaker
                speaker_gender = identify_speaker(paragraph, quote)
                chunks.append((speaker_gender, quote))
                start = end + len(quote)

            # Add remaining paragraph as narration
            if start < len(paragraph):
                chunks.append((('narrator', 'male'), paragraph[start:]))

    return chunks

def generate_audio(chunk, voice, index, temp_folder):
    speech_file_path = temp_folder / f"chunk_{index}.mp3"
    if not speech_file_path.exists():
        response = openai.audio.speech.create(
            model="tts-1-hd",
            voice=voice,
            input=chunk
        )
        response.stream_to_file(speech_file_path)
    return speech_file_path

def combine_audio(files, output_path):
    command = ["ffmpeg", "-y", "-i", "concat:" + "|".join(files), "-acodec", "copy", output_path]
    subprocess.run(command, check=True)

def main(book_file_path):
    logging.info(f"Starting processing of book: {book_file_path}")
    book_text = read_book(book_file_path)
    temp_folder = Path(__file__).parent / "temp_audio_files"
    temp_folder.mkdir(exist_ok=True)
    logging.info(f"Temporary folder created at {temp_folder}")

    voice_picker = assign_voice()
    chunks = chunk_text(book_text)
    audio_files = []

    for index, (speaker_gender, chunk) in enumerate(chunks):
        speaker, gender = speaker_gender
        voice = voice_picker(speaker, gender)
        logging.info(f"Generating audio for chunk {index} with speaker {speaker} and voice {voice}")
        audio_path = generate_audio(chunk, voice, index, temp_folder)
        audio_files.append(str(audio_path))

    combined_audio_path = Path(__file__).parent / "combined_speech.mp3"
    combine_audio(audio_files, str(combined_audio_path))
    logging.info(f"Audio files combined into {combined_audio_path}")

# Example usage
book_file_path = Path(__file__).parent / "book.txt"
main(book_file_path)
