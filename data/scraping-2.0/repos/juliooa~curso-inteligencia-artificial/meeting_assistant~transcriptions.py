import openai

def save_transcription(transcription, file_name):
    with open(file_name, "w") as text_file:
        text_file.write(transcription)

def open_transcription(file_name):
    with open(file_name, "r") as text_file:
        return text_file.read()
    
def transcribe_audio(audio_path):
    with open(audio_path, 'rb') as audio_file:
        transcription = openai.Audio.transcribe("whisper-1", audio_file)
    return transcription['text']