import openai
import re, os
import urllib.request
from gtts import gTTS
from moviepy.editor import *
from constants import *


# Set your OpenAI API key
openai.api_key = API_KEY

# Delete all files from output folders
all_files = [os.path.join(TEMP_DIRECTORY, file) for file in os.listdir(TEMP_DIRECTORY)]
for file in all_files:
    os.remove(file)

# Read the text file
with open(os.path.join(OUTPUT_DIRECTORY, "generated_text.txt"), "r") as file:
    text = file.read()

# Split the text by , and .
paragraphs = re.split(r"[.]", text)

clips = list()

# Loop through each paragraph and generate an image for each
for index, para in enumerate([p for p in paragraphs if len(p)>0]):
    print(f"Processing paragraph {index}...")
    response = openai.Image.create(
        prompt=para.strip(),
        n=1,
        size="1024x1024"
    )
    # Set temp location path
    temp_image_path = os.path.join(TEMP_DIRECTORY, "image{0}.jpg".format(index))
    temp_audio_path = os.path.join(TEMP_DIRECTORY, "voiceover{0}.mp3".format(index))
    
    # Generate New AI Image From Paragraph
    image_url = response['data'][0]['url']
    urllib.request.urlretrieve(image_url, temp_image_path)

    # Create gTTS instance and save to a file
    tts = gTTS(text=para, lang='en', slow=False)
    tts.save(temp_audio_path)

    # Load the audio file using moviepy
    audio_clip = AudioFileClip(temp_audio_path)
    audio_duration = audio_clip.duration

    # Load the image file using moviepy
    image_clip = ImageClip(temp_image_path).set_duration(audio_duration)

    # Use moviepy to create a text clip from the text
    text_clip = TextClip(para, fontsize=50, color="white")
    text_clip = text_clip.set_position('center').set_duration(audio_duration)

    # Use moviepy to create a final video by concatenating
    # the audio, image, and text clips
    clip = image_clip.set_audio(audio_clip)
    clips.append(CompositeVideoClip([clip, text_clip]))   

print("Concatenate All The Clips to Create a Final Video...")
final_video = concatenate_videoclips(clips, method="compose")
final_video = final_video.write_videofile(os.path.join(OUTPUT_DIRECTORY, "final_video.mp4"), fps=30)
print("The Final Video Has Been Created Successfully!")