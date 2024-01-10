import confidential 

from moviepy.editor import *
import random
import math
import openai
import requests
import re
from elevenlabs import *

set_api_key(confidential.elevenlabs_api_key)

Biden_voice = confidential.Biden_Voice

Trump_voice = confidential.Trump_Voice

Obama_voice = confidential.Obama_Voice

openai.api_key = confidential.openai_api_key

def func(description, num):

    text = "Transcribe a humorous skit between Trump, Biden, and Obama, where each of them are " + description + "\n\n  Only list their names and what they say, like this: “Obama: What do you think happened?” “Trump: I don't know.” Don’t give any stage directions or narration. Just their names and their dialogue. Don't have actions in asterisks or parentheses. Don't include any other characters. All lines must start with either “Trump: ” or “Obama: ” or “Biden: ”."
    
    while True:
        try:
            response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo-16k",
                messages = [{"role" : "user", "content": text}],
                temperature = 0.8,
                max_tokens = 12000,
            )
        except openai.error.RateLimitError:
            print("Rate Limit Error")
            continue
        except openai.error.InvalidRequestError:
            print("Invalid Request Error (too many tokens)")
            most_tokens -= 100
            continue
        except openai.error.APIError:
            print("API Error: probably bad gateway")
            continue
        except openai.error.ServiceUnavailableError:
                print("openai.error.ServiceUnavailableError")
                continue
        break

    script = response.choices[0].message["content"]

    script = script.replace("\n\n", "\n")
    script = re.sub(r'\([^)]*\)', '', script)
    script = re.sub(r'\*[^)]*\*', '', script)

    text = "Give a one-sentence summary of this chat in the structure of “The Presidents talk about” followed by the main topic of discussion in the following excerpt: " + script
     
    while True:
        try:
            response = openai.ChatCompletion.create(
                model = "gpt-3.5-turbo-16k",
                messages = [{"role" : "user", "content": text}],
                temperature = 0.8,
                max_tokens = 12000,
            )
        except openai.error.RateLimitError:
                print("Rate Limit Error")
                continue
        except openai.error.InvalidRequestError:
            print("Invalid Request Error (too many tokens)")
            most_tokens -= 100
            continue
        except openai.error.APIError:
            print("API Error: probably bad gateway")
            continue
        except openai.error.ServiceUnavailableError:
                print("openai.error.ServiceUnavailableError")
                continue
        break

    caption = response.choices[0].message["content"] 

    file1 = open("summary.txt", "a")
    file1.writelines(["Story" + str(num) + ": ", caption , "\n", "\n", script, "\n", "\n", description, "\n", "\n", "\n", "\n"])
    file1.close()
    
    while True:
        try:
            response = openai.Image.create_edit(
                image=open(confidential.Trump_Image, "rb"),
                mask=open(confidential.Trump_Mask, "rb"),
                prompt=description,
                n=1,
                size="1024x1024"
            )
        except openai.error.RateLimitError:
            print("Rate Limit Error")
            continue
        except openai.error.InvalidRequestError:
            print("Invalid Request Error (too many tokens)")
            most_tokens -= 100
            continue
        except openai.error.APIError:
            print("API Error: probably bad gateway")
            continue
        except openai.error.ServiceUnavailableError:
                print("openai.error.ServiceUnavailableError")
                continue
        break

    image_url = response['data'][0]['url']

    img_data = requests.get(image_url).content

    with open(confidential.Trump_Save_Name + str(num) + ".png", 'bw') as handler:
        handler.write(img_data)

    while True:
        try:
            response = openai.Image.create_edit(
                image=open(confidential.Obama_Image, "rb"),
                mask=open(confidential.Obama_Mask, "rb"),
                prompt=description,
                n=1,
                size="1024x1024"
            )
        except openai.error.RateLimitError:
            print("Rate Limit Error")
            continue
        except openai.error.InvalidRequestError:
            print("Invalid Request Error (too many tokens)")
            most_tokens -= 100
            continue
        except openai.error.APIError:
            print("API Error: probably bad gateway")
            continue
        except openai.error.ServiceUnavailableError:
                print("openai.error.ServiceUnavailableError")
                continue
        break

    image_url = response['data'][0]['url']

    img_data = requests.get(image_url).content

    with open(confidential.Obama_Save_Name + str(num) + ".png", 'bw') as handler:
        handler.write(img_data)

    while True:
        try:
            response = openai.Image.create_edit(
                image=open(confidential.Biden_Image, "rb"),
                mask=open(confidential.Biden_Mask, "rb"),
                prompt=description,
                n=1,
                size="1024x1024"
            )
        except openai.error.RateLimitError:
            print("Rate Limit Error")
            continue
        except openai.error.InvalidRequestError:
            print("Invalid Request Error (too many tokens)")
            most_tokens -= 100
            continue
        except openai.error.APIError:
            print("API Error: probably bad gateway")
            continue
        except openai.error.ServiceUnavailableError:
                print("openai.error.ServiceUnavailableError")
                continue
        break

    image_url = response['data'][0]['url']

    img_data = requests.get(image_url).content

    with open(confidential.Biden_Save_Name + str(num) + ".png", 'bw') as handler:
        handler.write(img_data)

    clip3 = ImageClip(confidential.Biden_Save_Name + str(num) + ".png")
    clip3 = clip3.resize(width = 720)
    difference_width = clip3.size[0] - 720
    difference_height = clip3.size[1] - 640
    Biden_Face = clip3.crop(x1=difference_width/2, y1 = difference_height/2, x2=clip3.size[0]-difference_width/2, y2 = clip3.size[1]-difference_height/2)

    clip4 = ImageClip(confidential.Obama_Save_Name + str(num) + ".png")
    clip4 = clip4.resize(width = 720)
    difference_width = clip4.size[0] - 720
    difference_height = clip4.size[1] - 640
    Obama_Face = clip4.crop(x1=difference_width/2, y1 = difference_height/2, x2=clip4.size[0]-difference_width/2, y2 = clip4.size[1]-difference_height/2)

    clip5 = ImageClip(confidential.Trump_Save_Name + str(num) + ".png")
    clip5 = clip5.resize(width = 720)
    difference_width = clip5.size[0] - 720
    difference_height = clip5.size[1] - 640
    Trump_Face = clip5.crop(x1=difference_width/2, y1 = difference_height/2, x2=clip5.size[0]-difference_width/2, y2 = clip5.size[1]-difference_height/2)

    trump_count = 0
    biden_count = 0
    obama_count = 0
    for i in script.split("\n"):
        try:
            if i.split(": ")[0] == "Trump":
                voice = Trump_voice
                filename = confidential.Trump_Audio + str(trump_count)
                trump_count += 1
                audio = generate(text = i.split(": ")[1], voice=voice)
                with open(filename + ".wav", mode='bw') as f:
                    f.write(audio)
            if i.split(": ")[0] == "Obama":
                voice = Obama_voice
                filename = confidential.Obama_Audio + str(obama_count)
                obama_count += 1
                audio = generate(text = i.split(": ")[1], voice=voice)
                with open(filename + ".wav", mode='bw') as f:
                    f.write(audio)
            if i.split(": ")[0] == "Biden":
                voice = Biden_voice
                filename = confidential.Biden_Audio + str(biden_count)
                biden_count += 1
                audio = generate(text = i.split(": ")[1], voice=voice)
                with open(filename + ".wav", mode='bw') as f:
                    f.write(audio)
        except (IndexError):
            print("Index Error")


    total_audio_duration = 0
    trump_count = 0
    biden_count = 0
    obama_count = 0
    for i in script.split("\n"):
        try:
            if i.split(":")[0] == "Trump":
                filename = confidential.Trump_Audio + str(trump_count) + ".wav"
                audio_clip = AudioFileClip(filename)
                total_audio_duration += audio_clip.duration
                trump_face_new = Trump_Face.set_duration(audio_clip.duration)
                trump_face_new = trump_face_new.set_audio(audio_clip)
                if trump_count == 0 and biden_count == 0 and obama_count == 0:
                    complete_video = trump_face_new
                else:
                    complete_video = concatenate_videoclips([complete_video, trump_face_new])
                trump_count += 1
            if i.split(":")[0] == "Obama":
                filename = confidential.Obama_Audio + str(obama_count) + ".wav"
                audio_clip = AudioFileClip(filename)
                total_audio_duration += audio_clip.duration
                obama_face_new = Obama_Face.set_duration(audio_clip.duration)
                obama_face_new = obama_face_new.set_audio(audio_clip)
                if trump_count == 0 and biden_count == 0 and obama_count == 0:
                    complete_video = obama_face_new
                else:
                    complete_video = concatenate_videoclips([complete_video, obama_face_new])
                obama_count += 1
            if i.split(":")[0] == "Biden":
                filename = confidential.Biden_Audio + str(biden_count) + ".wav"
                audio_clip = AudioFileClip(filename)
                total_audio_duration += audio_clip.duration
                biden_face_new = Biden_Face.set_duration(audio_clip.duration)
                biden_face_new = biden_face_new.set_audio(audio_clip)
                if trump_count == 0 and biden_count == 0 and obama_count == 0:
                    complete_video = biden_face_new
                else:
                    complete_video = concatenate_videoclips([complete_video, biden_face_new])
                biden_count += 1
        except (IndexError):
            print("Index Error")

    secondary_video = confidential.secondary_video_list

    while True:
        clip2 = VideoFileClip(secondary_video[random.randint(0, len(secondary_video) - 1)])
        if clip2.duration < total_audio_duration:
            continue
        break
    clip2 = clip2.without_audio()
    clip2 = clip2.resize(height = 640, width = 720)
    difference_width = clip2.size[0] - 720
    difference_height = clip2.size[1] - 640
    clip2 = clip2.crop(x1=difference_width/2, y1 = difference_height/2, x2=clip2.size[0]-difference_width/2, y2 = clip2.size[1]-difference_height/2)
    rand_start = random.randint(0, math.floor(clip2.duration - total_audio_duration - 1))
    clip2 = clip2.subclip(rand_start, rand_start+total_audio_duration)

    super_final_video = clips_array([[complete_video], [clip2]])

    super_final_video.write_videofile(confidential.video_location + "video" + str(num) + ".mp4",
                        codec='libx264', 
                        audio_codec='aac', 
                        temp_audiofile='temp-audio.m4a', 
                        remove_temp=True, 
                        fps=30)

def main():
    choice = input('Would you like to use your own prompt? (Y/N) ')

    if choice == "Y":
        description = input("Give a numbered list of prompts like so “1. A pirate in a pirate ship. 2. A cowboy in a saloon.”\n")

    if choice == "N":
        variable = input('How many videos would you like? ')
        text = "Give a list of " + str(variable) + " protagonist types and the environment they're in. For example, “A cowboy in a saloon.” or “A pirate in a pirate ship.” or “A knight in a castle.”"
        while True:
            try:
                response = openai.ChatCompletion.create(
                    model = "gpt-3.5-turbo-16k",
                    messages = [{"role" : "user", "content": text}],
                    temperature = 0.8,
                    max_tokens = 12000,
                )
            except openai.error.RateLimitError:
                print("Rate Limit Error")
                continue
            except openai.error.InvalidRequestError:
                print("Invalid Request Error (too many tokens)")
                most_tokens -= 100
                continue
            except openai.error.APIError:
                print("API Error: probably bad gateway")
                continue
            break
        description = response.choices[0].message["content"]
        
    if choice == "Y" or choice == "N":
        j = 0
        for i in re.split("[0-9]+\.", description):
            if(len(i) > 1 and i[1] == "A"):
                j += 1
                func(i, j)

main()