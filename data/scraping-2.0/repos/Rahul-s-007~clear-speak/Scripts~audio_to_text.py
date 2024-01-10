import openai
# import os
# from dotenv import load_dotenv
# load_dotenv()


#audio_file= open(f"{AUDIO_PATH}/speech.wav", "rb")
def transcript(ai_key,org_key):
    openai.api_key = ai_key
    openai.organization = org_key
    audio_file= open("speech.wav", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    #print(transcript)
    print(transcript['text'])
    return transcript['text']