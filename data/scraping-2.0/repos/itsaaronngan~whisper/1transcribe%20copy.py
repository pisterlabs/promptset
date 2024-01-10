import os
from pydub import AudioSegment
from openai import OpenAI
import tempfile
import sys

# Set up OpenAI client with your API key
client = OpenAI()

def split_audio(file_path):
    # Split the audio file into chunks
    audio = AudioSegment.from_file(file_path)
    # Calculate the chunk length in milliseconds based on the target file size
    chunk_length_ms = int((15000 * 50 * 1024 * 1024) / (audio.frame_rate * audio.frame_width * audio.channels))
    return [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

def transcribe_audio_chunks(chunks):
    transcripts = []
    temp_files = []  # List to store the locations of the temporary files
    for i, chunk in enumerate(chunks[:5]):  # Only process the first 4 chunks
        # Save the chunk as a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp:
            try:
                chunk.export(temp.name, format="mp3", bitrate="64k")
                print(f"Created temporary file: {temp.name}")
                temp_files.append(temp.name)  # Add the location of the temporary file to the list
            except Exception as e:
                print(f"Error creating temporary file: {e}")
                continue

            # Transcribe the temporary file
            try:
                audio_file=open(temp.name, "rb")
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,            
                )
                transcripts.append(transcript)
                print("Successful transcription")
            except Exception as e:
                print(f"Error transcribing audio: {e}")

        if i < len(chunks[:5]) - 1:
            print("Creating and uploading next chunk...")
    
    # Print the locations of the temporary files
    print("Locations of the temporary files:")
    for file in temp_files:
        print(file)

    return transcripts

def export_transcriptions(file_path, transcriptions, temp_files):
    # Export the transcriptions to a text file with the same name as the audio file
    audio_file_name = os.path.basename(file_path)
    text_file_name = "TRANSCRIBE_" + os.path.splitext(audio_file_name)[0].replace(".", "_") + ".txt"
    output_directory = "/Users/aaronnganm1/Documents/Coding/Whisper Transcription/output"
    text_file_path = os.path.join(output_directory, text_file_name)
    with open(text_file_path, "w") as file:
        for transcription in transcriptions:
            file.write(transcription.text + "\n")  # Access the 'text' attribute of the Transcription object
    print(f"Transcriptions saved at: {text_file_path}")

    # Delete the temporary files
    for temp_file in temp_files:
        os.remove(temp_file)
        print(f"Temporary file {temp_file} has been deleted")

def main(file_path):
    chunks, temp_files = split_audio(file_path)
    transcriptions = transcribe_audio_chunks(chunks)
    export_transcriptions(file_path, transcriptions, temp_files)

# Call the main function with the path to your audio file
if len(sys.argv) > 1:
    main(sys.argv[1])
else:
    print("No audio file provided.")
