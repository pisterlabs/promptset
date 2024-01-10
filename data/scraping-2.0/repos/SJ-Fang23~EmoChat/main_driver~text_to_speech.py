"""
text to speech using huggingface transdormers
"""
from pathlib import Path
from openai import OpenAI
import soundfile as sf
import sounddevice as sd
import time
from ctypes import *

robot_intro = "Hi, I am Misty. I am an experimental robot trying to learn more about humans and their daily activities. Tell me about something that has been bothering you lately."
speech_file_path = Path(__file__).parent.parent / "speech.mp3"
client = OpenAI()

ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
def py_error_handler(filename, line, function, err, fmt):
    pass
#   print ('messages are yummy')
c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)

asound = cdll.LoadLibrary('libasound.so')
# Set error handler
asound.snd_lib_error_set_handler(c_error_handler)

def text_to_speech(speech: str):
    response = client.audio.speech.create(
    model="tts-1",
    voice="nova",
    input=speech
    )
    response.stream_to_file(speech_file_path)
    data, fs = sf.read('speech.mp3',dtype='float32')
    sd.play(data,fs)
    status=sd.wait()

if __name__ == "__main__":
    text_to_speech(robot_intro)
    time.sleep(2)
    text_to_speech("The weather is really good!")
    

