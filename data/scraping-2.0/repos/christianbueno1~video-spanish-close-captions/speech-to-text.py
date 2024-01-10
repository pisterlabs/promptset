from openai import OpenAI
import glob
import os

openai_api_key = os.environ.get('OPENAI_API_KEY')

client = OpenAI()
audio_names = 'miracle-blind.mp3'
# audio_names = '*.mp3'

mp3_files = glob.glob(f'./audios/{audio_names}')

for file in mp3_files:
    print(file)
    print(f'Creating the transcription for the file: {file}')
    audio_file= open(file, "rb")
    transcript = client.audio.transcriptions.create(
    model="whisper-1", 
    file=audio_file,
    response_format="srt"
    )
    transcription_srt = transcript
    
    # print(type(transcript) )
    # print(transcript)
    print(transcription_srt)

    print(f"creating the transcription file name")
    transcription_filename = os.path.basename(file).split('.')[0] + '.srt'
    # print(transcription_filename)
    
    # Create the 'transcriptions' directory if it doesn't exist
    print(f" creating the directory fi it doesn't exist")
    if not os.path.exists('transcriptions'):
        os.makedirs('transcriptions')

    transcription_filepath = os.path.join('transcriptions', transcription_filename)
    # print(transcription_filepath)

    print(f"writing the transcription to the file")
    with open(transcription_filepath, 'w') as f:
        f.write(transcription_srt)


