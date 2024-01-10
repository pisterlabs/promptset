from dotenv import load_dotenv
load_dotenv()

import openai
import json
from pydub import AudioSegment

def split_audio(sizes=10):
    sound = AudioSegment.from_mp3("output.mp3")
    point_split = len(sound) // sizes

    for i in range(sizes):
        sound_segment = sound[point_split*i : point_split * (i + 1)]
        sound_segment.export("temp.mp3", format="mp3")
        yield "temp.mp3"

def get_transcript():
    for index, i in enumerate(split_audio()):
        print(f"index == {index}")
        audio_file = open(i, "rb")
        transcript = openai.Audio.transcribe("whisper-1", audio_file)
        with open(f"transcript_{index}.json", "w") as file:
            file.write(json.dumps(transcript))

if __name__ == "__main__":
    get_transcript()
