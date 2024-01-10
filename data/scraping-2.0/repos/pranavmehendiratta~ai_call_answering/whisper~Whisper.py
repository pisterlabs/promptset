import openai

def transcribeAudioFile(filename):
    file = open(filename, "rb")
    transcript = openai.Audio.transcribe("whisper-1", file)
    return transcript["text"]