import openai
import os

def stt(wav_path):
    openai.api_key=os.getenv("OPENAI_API_KEY")
    audio_file= open(f"{wav_path}", "rb")
    transc = openai.Audio.transcribe("whisper-1", audio_file)
    return transc.text

# # print(stt("server/audio_test.wav").text)
# with open("server/data/extracted.txt","w") as f:
#     f.write(stt("server/data/audio_ext.wav").text)
# f.close()
