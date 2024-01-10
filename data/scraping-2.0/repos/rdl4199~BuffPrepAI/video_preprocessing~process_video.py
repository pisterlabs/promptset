

import moviepy.editor as mp
import requests
import openai
openai.api_key = "Insert your openai api key"


def generate_subtitles(filename, seconds = 100, length = 769):
    i = 0
    while True:
        start, end = (i*seconds), (i*seconds + seconds)
        if end > length:
            break
        i += 1
        
        video_intro = 'vid.mp4'
        audio_intro = 'aud.mp3'

        logger = None
        # Uncomment below to see progress bar when saving
        logger='bar' 

        video = mp.VideoFileClip(filename)

        # # Clip a small section
        video_clip = video.subclip(start, end)
        # video_clip = video

        # Save audio
        video_clip.audio.write_audiofile(audio_intro, logger=None)

        # Save video
        # Doesn't appear to save the audio otherwise
        video_clip.write_videofile(video_intro,
                            codec='libx264', 
                            audio_codec='aac', 
                            temp_audiofile='temp-audio.m4a', 
                            remove_temp=True,
                            logger=logger 
                            )

        result = openai.Audio.transcribe(
            model='whisper-1',
            file=open(audio_intro, 'rb')
        )

        # print()
        with open("subtitles.txt", "a") as myfile:
            myfile.write(result['text']+str('\n'))

def generate_questions(subtitle_file_name):
    with open(subtitle_file_name, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            print(line)
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "system", "content": "You are a questioning assistant."},
                        {"role": "user", "content": line + "\n Write a question about this passage."}
                    ]
                )
            # print(response["choices"][0]["message"]["content"])
            with open("questions.txt", "a") as myfile:
                myfile.write(response["choices"][0]["message"]["content"]+str('\n'))

def summarize_data(subtitle_file_name):
    with open(subtitle_file_name, 'r') as f:
        while True:
            line = f.readline()
            if not line:
                break
            print(line)
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                        {"role": "system", "content": "You are a summarizing assistant."},
                        {"role": "user", "content": line + "\n Summarize this passage in 2 lines."}
                    ]
                )
            # print(response["choices"][0]["message"]["content"])
            with open("summarize.txt", "a") as myfile:
                myfile.write(response["choices"][0]["message"]["content"]+str('\n'))


if __name__ == "__main__":

    generate_subtitles(filename = "TED_Talk.mp4")
    generate_questions(subtitle_file_name = "subtitles.txt")
    summarize_data(subtitle_file_name = "subtitles.txt")