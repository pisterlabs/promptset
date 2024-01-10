import openai
import sounddevice as sd
from scipy.io.wavfile import write
from gtts import gTTS
from elevenlabs import generate, set_api_key, save
from elevenlabs.api import History

set_api_key("06dadcb30178b8ad0e19f25598461771")

import multiprocessing
import pyttsx3
import keyboard

def say(text):
		p = multiprocessing.Process(target=pyttsx3.speak, args=(text,))
		p.start()
		while p.is_alive():
			if keyboard.is_pressed('enter'):
				p.terminate()
			else:
				continue
		p.join()


def record_audio(filename, sec, sr = 44100):
    audio = sd.rec(int(sec * sr), samplerate=sr, channels=2, blocking=False)
    sd.wait()
    write(filename, sr, audio)

def record_audio_manual(filename, sr = 44100):
    input("  ** Press enter to start recording **")
    audio = sd.rec(int(10 * sr), samplerate=sr, channels=2)
    input("  ** Press enter to stop recording **")
    sd.stop()
    write(filename, sr, audio)

def play_audio(filename):
    signal, sr = af.read(filename)
    sd.play(signal, sr)

def transcribe_audio(filename):
    audio_file= open(filename, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    audio_file.close()
    return transcript

def translate_audio(filename):
    audio_file= open(filename, "rb")
    translation = openai.Audio.translate("whisper-1", audio_file)
    audio_file.close()
    return translation

def save_text_as_audio(text, audio_filename):
    print("SAVE TEXT AS AUDIO")
    print(text)
    audio = generate(
    text=text,
    voice="KIM23JcEVYJOlBQVOuBR",
    model='eleven_monolingual_v1'
)
    save(audio,audio_filename)

    #myobj = gTTS(text=text, lang='en', slow=False)  
   # myobj.save(audio_filename)



