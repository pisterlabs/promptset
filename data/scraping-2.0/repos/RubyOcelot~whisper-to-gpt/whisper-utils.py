import os
import sys
import openai
from audio_input_divide import split_audio

openai.api_key = os.getenv("OPENAI_API_KEY")

def audio_file_to_text(audio_file_name):
    audio_file = open(audio_file_name, "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file)
    return transcript

def print_text(transcript):
    text = transcript["text"]
    print(text)

if __name__ == "__main__":
    audio_file_name = sys.argv[1]
    output_file_base_name = "tmp/"+audio_file_name.split(".")[0]
    split_duration = 10
    split_num = 10
    split_audio(audio_file_name, split_duration, split_num)
    with open(output_file_base_name+"_split_"+str(split_duration)+"s_"+str(split_num)+".txt", "w") as f:
        for i in range(0, 10):
            text = audio_file_to_text(output_file_base_name+str(split_duration)+"s_"+str(i)+".mp3")["text"]
            f.write(text)

    split_audio(audio_file_name, split_duration*split_num, 1)
    with open(output_file_base_name+"_whole"+".txt", "w") as f:
        f.write(audio_file_to_text(audio_file_name)["text"])
