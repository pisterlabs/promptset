import openai
import argparse

def transcribe_audio_with_openai(audio_file):
    client = openai.OpenAI()
    
    with open(audio_file, "rb") as audio:
        # Call the OpenAI API to transcribe the audio file
        # OpenAI Python client library handles the file upload
        transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=audio
        )
        
    return transcript.text


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', help='Path to the audio file')
    args = parser.parse_args()
    
    if args.file:
        audio_file = args.file
    else:
        print("Please provide the path to the audio file using the --file flag.")
        exit(1)
    
    # Transcribe the audio file using OpenAI API
    transcript = transcribe_audio_with_openai(audio_file)
    
    # Print the transcribed audio
    print(transcript)
