from decouple import config
from openai import OpenAI

api_key = config("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

class STT:
    def __init__(self, filename=None):
        self.filename = filename
        self.audio_file = open(self.filename, "rb")

    def transcribe(self):
        if self.filename is not None:
            transcript = client.audio.transcriptions.create(
            model="whisper-1", 
            file=self.audio_file, 
            response_format="text"
            )
            return transcript

def main():    
    # arquivo só para teste, excluir dps
    stt = STT(filename='./audio/audio.ogg')
    speech_text = stt.transcribe()
    print(speech_text)

if __name__ == "__main__":
    main()