import os
from openai import OpenAI
import sys
client = OpenAI(
            api_key=os.getenv("OPEN_API_KEY")
        )
video_id = sys.argv[1]

audio_file_path = os.path.join(os.getcwd(), 'tmp', video_id + '.m4a')

audio_file = open(audio_file_path, 'rb')
transcript = client.audio.transcriptions.create(
        model="whisper-1",
        file=audio_file,
        response_format="srt",
        prompt="Iam a programmer. This is a project Iam working on"
        )
print(transcript)
