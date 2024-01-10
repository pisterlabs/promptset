import keyboard
import requests
import json
import openai
from PIL import Image, ImageDraw, ImageFont, ImageOps
import textwrap
import random
import os
import time
from bing_image_downloader import downloader
from moviepy.editor import *
from bing_image_downloader import downloader
import openai
import os
import keyboard

openai.api_key = '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'

cop = input("enter content:")
top = input("enter image:")
num = int(input("num:"))
# Replace spaces in title with hyphens

# Download images of dogs
downloader.download(f"{top}", limit=num, output_dir="images",
                    adult_filter_off=True, force_replace=False)

for i in range(1, num+1):
    with open("news.txt", "r") as file:
        news = file.read().strip()
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": f"write a motivation line on {cop.upper()} , dont use : {news}  just write a motivation line"}
        ]
    )
    OP = response['choices'][0]['message']['content']
    OP = OP.replace('"', '')

    # Make a request to the Unsplash API to get a random photo

    image_dir = f"images/{top}/image_{i}"
    # Check if the image file exists with a .jpg or .png extension
    if os.path.exists(f"{image_dir}.jpg"):
        image_dir += ".jpg"
    elif os.path.exists(f"{image_dir}.png"):
        image_dir += ".png"
    else:
        print(f"No image found for {top} {i}")
        continue  # Skip to the next iteration if no image is found
        # Create an Image object with the local image file

        # Create an Image object with the local image file

    # Create an Image object with the local image file
    img = Image.open(image_dir)

    # Create a new white image of 600x600 size
    new_img = Image.new('RGB', (600, 600), (255, 255, 255))

    # Calculate the center position of the new image
    x = (600 - img.width) // 2
    y = (600 - img.height) // 2

    # Paste the original image onto the new image at the center position
    new_img.paste(img, (x, y))

    # Load the overlay image
    overlay = Image.open('tranf.png')
    overlay = overlay.resize((600, 600))
    overlay = overlay.convert("RGBA")

    # Paste the overlay image onto the main image
    new_img.paste(overlay, (0, 0), overlay)
    # Display the image

    # Create an Image object with the randomly chosen background color

    # Create a Draw object for drawing on the image
    draw = ImageDraw.Draw(new_img)
    # Create a black layer with transparency of 30%

    # reduced font size to 24
    font = ImageFont.truetype('Muroslant.otf', size=23)

    # Define the text to draw
    text = f"{OP}"

    # Wrap the text to fit within the image width
    wrapped_text = textwrap.wrap(text, width=50)

    # Calculate the position of the text in the center of the image
    x = (new_img.width - draw.textbbox((0, 0),
                                       wrapped_text[0], font=font)[2]) // 2
    y = (new_img.height -
         font.getsize(wrapped_text[0])[1] * len(wrapped_text)) // 2

    # Draw each line of wrapped text on the image
    for line in wrapped_text:
        draw.text((x, y), line, fill='white',
                  font=font, outline='black')
        # draw.text((x, y), line, fill='black', font=font, stroke_width=1, stroke_fill='white', align='center')
        y += draw.textsize(line, font=font)[1]

    # Get the news item's title
    title = f"{top}"

    # Create a folder to store the images if it doesn't exist
    if not os.path.exists(title):
        os.makedirs(title)

    # Save the image with a unique filename
    i = 1
    while True:
        img_filename = f"{title}/image{i}.png"
        if not os.path.exists(img_filename):
            break
        i += 1
    new_img.save(img_filename)
    print(f"Saved image as {img_filename}")

    # Set the dimensions of the video
    VIDEO_WIDTH = 720
    VIDEO_HEIGHT = 1280

    # Set the duration of each image
    INITIAL_IMAGE_DURATION = 15

    # Set the path to the music file
    MUSIC_PATH = "data.mp3"

    # Set the directory path to the folder containing the images
    folder_path = f"{top}/"

    # Get the file paths to all the images in the folder
    IMAGE_PATHS = [os.path.join(folder_path, f) for i in range(1, num+1) for f in os.listdir(
        folder_path) if f.endswith(('.jpg', '.png')) and f.startswith(f"image{i}")]

    clip_duration = 15

    # Create a list of video clips
    video_clips = []
    for image_path in IMAGE_PATHS:
        try:
            # Create an image clip for the current image
            image_clip = ImageClip(image_path)
            # Calculate the new height based on the aspect ratio of the original image
            new_height = int(VIDEO_WIDTH / image_clip.w * image_clip.h)
            # Resize the image to fit the video dimensions without black bars
            image_clip = image_clip.resize((VIDEO_WIDTH, new_height))
            image_clip = image_clip.set_position(("center", "center"))
            image_clip = image_clip.set_duration(clip_duration)

            # Create a black background clip
            bg_clip = ColorClip((VIDEO_WIDTH, VIDEO_HEIGHT), color=(0, 0, 0))
            bg_clip = bg_clip.set_duration(clip_duration)

            # Combine the image clip with the background clip
            video_clip = CompositeVideoClip([bg_clip, image_clip])

            # Append the video clip to the list
            video_clips.append(video_clip)
        except OSError as e:
            print(f"Error processing image {image_path}: {str(e)}")
            continue

    # Concatenate the video clips in a loop until the audio ends
    final_clip = concatenate_videoclips(video_clips, method="compose", bg_color=(
        0, 0, 0))

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
