from decouple import config
from openai import OpenAI
import sys
from playsound import playsound

api_key = config("OPENAI_API_KEY")

client = OpenAI(api_key=api_key)

class TTS:
    def __init__(self, filename=None, text=None):
        self.filename = filename
        self.text = text
    
    def transcript(self):
        """Gets the textual content of the provided element (e.g. filename's info or raw text)"""
        if self.filename is not None:
            with open(self.filename, "r") as f:
                self.text = f.read()
        
        return self.text
    
    def generate_audio(self):
        """Using the OpenAI API, transforms the current text into audio file"""
        self.transcript()
        response = client.audio.speech.create(
            model='tts-1',
            voice='alloy',
            input=self.text,
        )

        response.stream_to_file("./audio/audio.mp3")
        return

    def play(self):
        """Plays the generated audio"""
        playsound("audio.mp3")
        return


def main(args=sys.argv):
    tts = TTS(filename=None, text=args[1])
    tts.generate_audio()
    tts.play()


if __name__ == "__main__":
    main()