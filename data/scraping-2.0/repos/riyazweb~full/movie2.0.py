from moviepy.editor import *
from bing_image_downloader import downloader
import openai
from feautures.custom_voice import speak
import os
import keyboard

openai.api_key = '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'

top = input("Enter the title of video: ")


def cat():
    with open('news.txt', 'r') as f:
        content = f.read()

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": f"write about mr beast biography  in 400 words paragarph  in style of youtuber and dontt mention subscriber count and ask to subecribe at end of video"}]
        # messages=[{"role": "user", "content": f"write about orgianl mystery story of pyramids  in 300 words paragarph  in style of youtuber and ask to subecribe at end of video in hinglish text in englsh but language hindi and dont write any translation"}]

        # messages=[{"role": "user", "content": f"in hinglish write one mind blowing intrestng fact on {top} in hinglish text in english but language hindi  and deatils with only 70 words, dont write  any translataion"}]
    )

    OP = response['choices'][0]['message']['content']
    OP = OP.replace('"', '')

    print(OP)
    # speak("hello dosto" + " " + OP)
    speak("hello guys" + " " + OP)

cat()

# Set the dimensions of the video
VIDEO_WIDTH = 854
VIDEO_HEIGHT = 480

# Set the duration of each image
IMAGE_DURATION = 1.5

# Set the path to the music file
MUSIC_PATH = "data.mp3"

# Download images of cats
downloader.download(f"{top}", limit=9, output_dir="images",
                    adult_filter_off=True, force_replace=False)

# Set the directory path to the folder containing the images
folder_path = f"images/{top}/"

# Get the file paths to all the images in the folder
IMAGE_PATHS = [os.path.join(folder_path, f)
               for f in os.listdir(folder_path) if f.endswith('.jpg')]

num_images = len(IMAGE_PATHS)
audio_clip = AudioFileClip(MUSIC_PATH)
audio_duration = audio_clip.duration
IMAGE_DURATION = audio_duration / num_images

# Create a list of video clips
video_clips = []
for image_path in IMAGE_PATHS:
    # Create an image clip for the current image
    image_clip = ImageClip(image_path)
    # Calculate the new height based on the aspect ratio of the original image
    new_height = int(VIDEO_WIDTH / image_clip.w * image_clip.h)
    # Resize the image to fit the video dimensions without black bars
    image_clip = image_clip.resize((VIDEO_WIDTH, new_height))
    image_clip = image_clip.set_position(("center", "center"))
    image_clip = image_clip.set_duration(IMAGE_DURATION)

    # Create a black background clip
    bg_clip = ColorClip((VIDEO_WIDTH, VIDEO_HEIGHT), color=(0, 0, 0))
    bg_clip = bg_clip.set_duration(IMAGE_DURATION)

    # Combine the image clip with the background clip
    video_clip = CompositeVideoClip([bg_clip, image_clip])

    # Append the video clip to the list
    video_clips.append(video_clip)

# Concatenate the video clips in a loop until the audio ends
audio_clip = AudioFileClip(MUSIC_PATH)
audio_duration = audio_clip.duration
final_clip = concatenate_videoclips(video_clips, method="compose", bg_color=(
    0, 0, 0)).set_duration(audio_duration).loop(duration=audio_duration)

# Set the audio file for the final video clip
audio_clip = audio_clip.set_duration(final_clip.duration)
final_clip = final_clip.set_audio(audio_clip)

# Set the desired output file name
filename = f"{top}.mp4"

# Check if the file already exists
if os.path.isfile(filename):
    # If it does, add a number to the filename to create a unique name
    basename, extension = os.path.splitext(filename)
    i = 1
    while os.path.isfile(f"{basename}_{i}{extension}"):
        i += 1
    filename = f"{basename}_{i}{extension}"

# Write the video file with the updated filename
final_clip.write_videofile(filename, fps=30)
