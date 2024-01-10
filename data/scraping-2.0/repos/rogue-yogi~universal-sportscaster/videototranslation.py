import os
import openai
import requests
import io
import tempfile
import re
from pydub import AudioSegment

def transcribe_audio(file_path):
    audio = AudioSegment.from_file(file_path)
    chunk_length = 30 * 1000  # 30 seconds in milliseconds
    chunks = [audio[i:i + chunk_length] for i in range(0, len(audio), chunk_length)]
    full_transcription = ''
    for chunk in chunks:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        chunk.export(temp_file.name, format="wav")
        with open(temp_file.name, "rb") as file:
            transcript = openai.Audio.transcribe(model="whisper-1", file=file)
            full_transcription += transcript.text + ' '
        os.unlink(temp_file.name)
    return full_transcription

def translate_text(text, language):
    MAX_WORDS = 500
    translated_text = ''
    words = text.split()
    for i in range(0, len(words), MAX_WORDS):
        chunk = " ".join(words[i:i + MAX_WORDS])
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": f"You are a helpful assistant who translates English to {language}."},
                {"role": "user", "content": f"Translate the following from English to colloquial {language} but rephrase the metaphors and other language that doesn't translate directly into {language}. Convey the underlying message behind the passage, not just of the words themselves. Output ONLY {language} text, and NOT English:\n{chunk}"}
            ]
        )
        translated_text += completion.choices[0].message['content'].strip() + ' '
    return translated_text

def main():
    openai_api_key = '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'
    openai.api_key = openai_api_key
    language = 'Mandarin'
    video_parts_directory = '/Users/livestream/Desktop/synclabs/Chunks'
    
    output_file_name = "translations.txt"
    part_filenames = [f for f in os.listdir(video_parts_directory) if f.startswith('part') and f.endswith('.mp4')]

    # Sort filenames by the part number
    part_filenames.sort(key=lambda x: int(re.search(r'\d+', x).group()))

    with open(output_file_name, 'w') as output_file:
        part_number = 1
        for part_filename in part_filenames:
            video_path = os.path.join(video_parts_directory, part_filename)
            print(f"Transcribing audio from {part_filename}...")
            transcription = transcribe_audio(video_path)
            print("Transcription completed.")

            print("Translating text...")
            translation = translate_text(transcription, language)
            print("Translation completed.")

            # Write the entire translation without splitting
            output_file.write(f"Part {part_number}:\n")
            output_file.write(translation)
            output_file.write("\n\n")

            part_number += 1

    print(f"All translations saved to '{output_file_name}'.")

if __name__ == "__main__":
    main()