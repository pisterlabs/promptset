import openai
import re, os
import urllib.request
from gtts import gTTS
from moviepy.editor import *
from api_key import API_KEY
from moviepy.config import change_settings

change_settings({"IMAGEMAGICK_BINARY": r'C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\convert.exe'})
#
openai.api_key = API_KEY
with open("generated_text.txt", "r") as file:
    text = file.read()

# split the text by , and .
paragraphs = re.split(r"[,.]", text)

# create folders
os.makedirs("audio")
os.makedirs("images")
os.makedirs("video")

# loop through each
i = 1
for para in paragraphs[:-1]:
    response = openai.Image.create(
        prompt=para.strip(),
        n=1
        #size="1024x1024"
    )
    print("generate new img from para")
    image_url = response['data'][0]['url']
    urllib.request.urlretrieve(image_url, f"images/image{i}.jpg")
    print("generated image saved in img folder")

    # create gtts instance
    tts = gTTS(text=para, lang='en', slow=False)
    tts.save(f"audio/voiceover{i}.mp3")
    print("paragraph converted to voice")

    print("extract voice get duration")
    audio_clip = AudioFileClip(f"audio/voiceover{i}.mp3")
    audio_duration = audio_clip.duration

    # audio file using moviepy
    print("extract image clip and set duration")
    image_clip = ImageClip(f"images/image{i}.jpg").set_duration(audio_duration)

    print("customize text clip")
    text_clip = TextClip(para, fontsize=25, color="white")
    text_clip = text_clip.set_pos('center').set_duration(audio_duration)

    # use py to create final video
    print("concatenated video")
    clip = image_clip.set_audio(audio_clip)
    video = CompositeVideoClip([clip, text_clip])

    # save final video to file
    video = video.write_videofile(f"video/video{i}.mp4", fps=24)
    print(f"The Video{i} Has Been Created Successfully!")
    i += 1

clips = []
l_files = os.listdir("video")
for file in l_files:
    clip = VideoFileClip(f"video/{file}")
    clips.append(clip)

print("Concatenate All The Clips to Create a Final Video...")
final_video = concatenate_videoclips(clips, method="compose")
final_video = final_video.write_videofile("final_video.mp4")
print("The Final Video Has Been Created Successfully!")