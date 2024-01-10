# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
import moviepy.editor as mp
from config import OPENAI_API_KEY

# def separate():
#     my_clip = mp.VideoFileClip(r"video.mp4")
#     my_clip.audio.write_audiofile(r"audio.mp4")

def convert():
    openai.api_key = OPENAI_API_KEY

    # prompt = """
    # Please transcribe the following audio file:

    # include filler words and punctuation.
    # """

    # separate()

    audio_file= open("./video.mp4", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)

    f = open('transcript.txt', 'w')
    f.write(transcript.get("text"))
    f.close()