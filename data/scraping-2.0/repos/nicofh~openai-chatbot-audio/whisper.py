from dotenv import load_dotenv
from pydub import AudioSegment
import os
import openai
import datetime
import sys

sys.path.append('/path/to/ffmpeg')
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

class WhisperExporter:

    def __init__(self):
        self._textes = []

    def whisper_to_text(self, path: str, filename: str):
        f = open(path, "rb")
        transcript = openai.Audio.transcribe("whisper-1", file=f)
        self.textes.append({"filename": filename, "text": transcript["text"]})
        return f"transcribed {path}"

    @staticmethod
    def to_txt(text_list: list):
        for text in text_list:
            with open(f"./text_files/{text['filename']}.txt", "w") as f:
                f.write("".join(text['text']))
                f.close()
                print(f"saved to text_files/{text['filename']}")

    @property
    def textes(self):
        return self._textes



if __name__ == "__main__":
    exporter = WhisperExporter()
    # loop over all files if they end with the allowed endings
    for file in [f for f in os.listdir("./audio_files") if os.path.splitext(f)[1] in [".m4a", ".mp3", ".mp4", ".mpeg", ".mpga", ".wav", ".webm", ".ogg"]]:
        filename = os.path.splitext(file)[0]
        file_extension = os.path.splitext(file)[1]
        file_path = f"./audio_files/{file}"
        # convert .ogg files to .mp4
        if file_extension == ".ogg":
            new_file_path = f"./audio_files/{filename}.mp4"
            AudioSegment.from_ogg(file_path).export(new_file_path, format="mp4")
            file_path = new_file_path
            # DELETE OGG FILE HERE
        exporter.whisper_to_text(file_path, filename) #  THIS METHOD ADDS "." TO PATH
    exporter.to_txt(exporter.textes)
