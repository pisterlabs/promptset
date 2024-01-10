import openai

def promptToWhipser(file_path):

    audio_file = file_path
    # transcripting speech to audio with Whisper
    try:
        transcript = openai.Audio.transcribe_("whisper-1", audio_file)
        return transcript['text']
    except Exception as error:
        return "ERROR"