from lyricsgenius import Genius
from openai import OpenAI
from dotenv import load_dotenv
import requests
import os
import sys
import uuid
from slugify import slugify

# load environment variables
load_dotenv()

# set up OpenAI API
openai_api_key = os.getenv("OPENAI_API_KEY")

# set up Genius API
genius_api_key = os.getenv("GENIUS_API_KEY")

genius = Genius(genius_api_key)

genius.remove_section_headers = True # remove section headers (e.g. [Chorus]) from lyrics when searching

# set up OpenAI Client

client = OpenAI(api_key = openai_api_key)

# Set up function to generate images from lyrics

def images_from_lyrics(artist_name, song_name, lyrics):
    # create a unique directory name for the song with a UUID at the end
    dirname = f"{artist_name} - {song_name} - {uuid.uuid4()}"
    # create the directory if it doesn't exist
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    # iterate through each line of lyrics, keeping track of line number

    for i, lyric in enumerate(lyrics):
        # print status message
        print(f"Generating image for line {i+1} of {len(lyrics)}...")
        prompt = f"Artist: '{artist_name}'\nSong title: '{song_name}'\nLyric to visualize: '{lyric}'"
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

# input artist name

artist_name = input("Enter artist name: ")

# input song name

song_name = input("Enter song name: ")

# search for song

song = genius.search_song(song_name, artist_name)

# jsonify lyrics

lyrics = song.lyrics
lyrics = lyrics.split("\n")

# remove the first line of lyrics

lyrics = lyrics[1:]

# remove empty lyrics 

lyrics = list(filter(lambda x: x != "", lyrics))

# remove the string "22Embed" from the last line

lyrics[-1] = lyrics[-1][:-7]

# generate images from lyrics

images_from_lyrics(artist_name, song_name, lyrics)

# print success message

print(f"Images generated for {artist_name} - {song_name}!")