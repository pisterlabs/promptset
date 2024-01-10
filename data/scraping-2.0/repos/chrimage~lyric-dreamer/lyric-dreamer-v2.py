from lyricsgenius import Genius
from openai import OpenAI
from dotenv import load_dotenv
import requests
import os
import sys
import uuid
from slugify import slugify
import json

# load environment variables
load_dotenv()

# set up OpenAI API
openai_api_key = os.getenv("OPENAI_API_KEY")

# set up Genius API
genius_api_key = os.getenv("GENIUS_API_KEY")

genius = Genius(genius_api_key)

# remove section headers (e.g. [Chorus]) from lyrics when searching
genius.remove_section_headers = True

# set up OpenAI Client

client = OpenAI(api_key=openai_api_key)

# input artist name

artist_name = input("Enter artist name: ")

# input song name

song_name = input("Enter song name: ")

# search for song

song = genius.search_song(song_name, artist_name)

lyrics = song.lyrics

artist_name = song.artist

song_name = song.title

# come up with a concept for the video

response = client.chat.completions.create(
    model="gpt-3.5-turbo-1106",
    response_format={"type": "json_object"},
    temperature=1.1,
    messages=[
        {
            "role": "system",
            "content": "You are a creative assistant who helps come up with concepts for lyric image videos to accompany a song. Come up with a concept for a lyric image video to accompany the song. Avoid describing animated scenes, the tool we are using can only generate still images.",
        },
        {
            "role": "user",
            "content": f"Here are the lyrics for {song.title} by {song.artist}:\n{lyrics}\n Now, let's come up with 10 different possible concepts for the video. Remember not to describe anything animated. Return as a json object. Include the keys 'visual_style' and 'concept' in the object.",
        },
    ],
)

concepts = response.choices[0].message.content

concepts = json.loads(concepts)

concepts = concepts["concepts"]

# prompt the user to choose a concept

for i, concept in enumerate(concepts):
    print(f"{i+1}. {concept}")

concept_choice = input("Choose a concept: ")

chosen_concept = concepts[int(concept_choice) - 1]

# print the concept

print(
    f"Visual Style: {chosen_concept['visual_style']}\nConcept: {chosen_concept['concept']}"
)

visual_style = chosen_concept["visual_style"]

concept = chosen_concept["concept"]

# generate a list of image descriptions based on the visual style and concept

response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    response_format={"type": "json_object"},
    messages=[
        {
            "role": "system",
            "content": f"Generate a list of image descriptions based on the visual style '{visual_style}' and concept '{concept}'. Try to include a new image for each lyric, even if it is a repeat.",
        },
        {
            "role": "user",
            "content": f"Here are the lyrics for {song.title} by {song.artist}:\n{lyrics}\n Now, let's generate a list of image descriptions based on the visual style '{visual_style}' and concept '{concept}'. Return as json list of dicts. Use the keys 'lyric' and 'image_description' for each line of the song.",
        },
    ],
)

image_descriptions = response.choices[0].message.content

image_descriptions = json.loads(image_descriptions)

images = image_descriptions["images"]

# create a directory for the images

dirname = f"{artist_name}-{song_name}-{uuid.uuid4()}"

# create the directory if it doesn't exist

if not os.path.exists(dirname):
    os.mkdir(dirname)

# iterate through each image description
# image['lyric'] is the lyric to visualize
# image['image_description'] is the image description to use

for i, image in enumerate(images):
    # print status message
    print(f"Generating image for line {i+1} of {len(images)}...")
    lyric = image["lyric"]
    image_description = image["image_description"]
    prompt = f"Artist: '{artist_name}'\nSong title: '{song_name}'\nLyric to visualize: '{lyric}'\nImage description: '{image_description}'"
    # generate image from lyric, and handle errors
    try:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            style="vivid",
            n=1,
        )
    except KeyboardInterrupt:
        print("Keyboard interrupt. Exiting...")
        sys.exit()
    except:
        print("Error generating image. Skipping...")
        continue
    image_url = response.data[0].url
    # store the revised_prompt for the next iteration
    last_prompt = response.data[0].revised_prompt
    # download the image
    image = requests.get(image_url)
    # slugify the lyric
    lyric_slug = slugify(lyric)
    # save the image
    with open(f"{dirname}/{i}-{lyric_slug}.jpg", "wb") as f:
        f.write(image.content)
