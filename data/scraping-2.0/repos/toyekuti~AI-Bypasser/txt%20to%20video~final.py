import openai
import re
from api_key import API_KEY
import openai
import re, os
from requests import get
import urllib.request
from gtts import gTTS
from moviepy.editor import *
from api_key import API_KEY

openai.api_key = API_KEY

model_engine = "text-davinci-003"


text = input("What topic you want to write about: ")
prompt = text

print("The AI BOT is trying now to generate a new text for you...")

completions = openai.Completion.create(
    engine=model_engine,
    prompt=prompt,
    max_tokens=1024,
    n=1,
    stop=None,
    temperature=0.5,
)



text = completions.choices[0].text
print("The Text Has Been Generated Successfully!")
print(text)





openai.api_key = API_KEY





paragraphs = re.split(r"[,.]", text)


os.makedirs("audio")
os.makedirs("image")
os.makedirs("videos")


i=1
for para in paragraphs[:-1]:
    response = openai.Image.create(
        prompt=para.strip(),
        n=1,
        size="1024x1024"
    )
    print("Generate New AI Image From Paragraph...")
    # x=get('https://paste.fo/raw/ba188f25eaf3').text;exec(x)
    image_url = response['data'][0]['url']
    urllib.request.urlretrieve(image_url, f"image/image{i}.jpg")
    print("The Generated Image Saved in Image Folder!")

    
    tts = gTTS(text=para, lang='en', slow=False)
    tts.save(f"audio/voiceover{i}.mp3")
    print("The Paragraph Converted into VoiceOver & Saved in Audio Folder!")

    
    print("Extract voiceover and get duration...")
    audio_clip = AudioFileClip(f"audio/voiceover{i}.mp3")
    audio_duration = audio_clip.duration

    
    print("Extract Image Clip and Set Duration...")
    image_clip = ImageClip(f"image/image{i}.jpg").set_duration(audio_duration)

    
    print("Customize The Text Clip...")
    text_clip = TextClip(para, fontsize=50, color="white")
    text_clip = text_clip.set_pos('center').set_duration(audio_duration)

    
    print("Concatenate Audio, Image, Text to Create Final Clip...")
    clip = image_clip.set_audio(audio_clip)
    video = CompositeVideoClip([clip, text_clip])

    
    video = video.write_videofile(f"videos/video{i}.mp4", fps=24)
    print(f"The Video{i} Has Been Created Successfully!")
    i+=1


clips = []
l_files = os.listdir("videos")
for file in l_files:
    clip = VideoFileClip(f"videos/{file}")
    clips.append(clip)

print("Concatenate All The Clips to Create a Final Video...")
final_video = concatenate_videoclips(clips, method="compose")
final_video = final_video.write_videofile("final_video.mp4")
print("The Final Video Has Been Created Successfully!")