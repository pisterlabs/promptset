import sounddevice as sd
import soundfile as sf
import openai

def recording_input():
    sample_rate = 44100
    duration = 5  
    channels = 1  
    print("Recording Audio...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels)
    sd.wait() 
    output_file = "Input_Audio/output.mp3"  # Output file path
    sf.write(output_file, recording, sample_rate)
    audio_file = open("Input_Audio/output.mp3", "rb")
    transcript = openai.Audio.transcribe(
        model ="whisper-1", file= audio_file, temperature = 0.2)
    print(transcript['text'])
    return transcript['text']


