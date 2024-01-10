from dotenv import load_dotenv
from langchain.llms import VertexAI
from langchain.chat_models import ChatVertexAI
from langchain.schema.messages import HumanMessage
from google.cloud import texttospeech

import spotipy
from spotipy.oauth2 import SpotifyOAuth
import pygame
import os
from PIL import Image
from PIL import ImageOps as PIL_ImageOps
from PIL.ExifTags import TAGS, GPSTAGS
import googlemaps
import json
from vertexai.preview.generative_models import GenerativeModel, GenerationResponse, GenerationConfig


load_dotenv()
if __name__ == "__main__":
    pass

verbose = True
temp = 0
finish = 0

def get_image_metadata(image_path):
    # Open the image
    image = Image.open(image_path)

    # Extract the EXIF data
    exif_data = image._getexif()
    if exif_data is not None:
        # Translate the EXIF data to labelled data
        labelled_exif_data = {TAGS[key]: exif_data[key] for key in exif_data.keys() if key in TAGS and isinstance(exif_data[key], (bytes, str))}
        return labelled_exif_data
    else:
        return "No metadata found"
def get_geotagging(image_path):
    image = Image.open(image_path)
    exif = image._getexif()

    if not exif:
        raise ValueError("No EXIF metadata found")

    geotagging = {}
    for (idx, tag) in TAGS.items():
        if tag == 'GPSInfo':
            if idx not in exif:
                raise ValueError("No EXIF geotagging found")

            for (t, value) in GPSTAGS.items():
                if t in exif[idx]:
                    geotagging[value] = exif[idx][t]

    return geotagging

def get_location_by_coordinates(lat, lon):
    gmaps = googlemaps.Client(key=os.environ.get("GOOGLE_MAPS_API_KEY"))
    reverse_geocode_result = gmaps.reverse_geocode((lat, lon))
    locality = ""
    country = ""
    for component in reverse_geocode_result[0]['address_components']:
        if 'locality' in component['types']:
            locality = component['long_name']
        if 'country' in component['types']:
            country = component['long_name']
    

    address = locality + ", " + country
    return address

def get_decimal_from_dms(dms, ref):
    degrees, minutes, seconds = dms
    result = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ['S', 'W']:
        result = -result
    return result

def get_text(response: GenerationResponse):
  """Returns the Text from the Generation Response object."""
  part = response.candidates[0].content.parts[0].text
  return part

def retrieve_song_title(query,temp: int, address: str):
    # llm = VertexAI( model_name="text-bison-32k", temperature=temp, max_output_tokens=8100)
    model = GenerativeModel("gemini-pro")
    json_format = {
        "song_name": "name of the song",
        "author": "author of the song"
    }
    prompt = f"Based on this description, name me a song I can play to feel this photo taken in {address}. The song should reflect the mood of the image and taking into account the address.\nDescription: {query}. Use the following json format: {json_format} using double quotes."
    
    generation_config = GenerationConfig(
        temperature=temp,
        top_p=1.0,
        top_k=32,
        candidate_count=1,
        max_output_tokens=8192,
    )
    
    responses = model.generate_content(prompt, stream=False,generation_config=generation_config)
    return responses.candidates[0].content.parts[0].text   

def play_mp3(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)

# Call the function with the path to your MP3 file
def text_to_speech(text, filename):
    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)

    # Note: the voice can also be specified by language code and name
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Studio-O",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE ,
    )

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )

    # Write the response to the output file.
    with open(filename, "wb") as out:
        out.write(response.audio_content)
        print(f'Audio content written to file "{filename}"')  
    play_mp3(filename)
# def text_to_speech_ssml(text, filename):
#     client = texttospeech.TextToSpeechClient()

#     input_text = texttospeech.SynthesisInput(ssml=text)

#     # Note: the voice can also be specified by language code and name
#     voice = texttospeech.VoiceSelectionParams(
#         language_code="en-US",
#         name="en-US-Wavenet-F",
#         ssml_gender=texttospeech.SsmlVoiceGender.FEMALE ,
#     )

#     audio_config = texttospeech.AudioConfig(
#         audio_encoding=texttospeech.AudioEncoding.MP3
#     )

#     response = client.synthesize_speech(
#         request={"input": input_text, "voice": voice, "audio_config": audio_config}
#     )

#     # Write the response to the output file.
#     with open(filename, "wb") as out:
#         out.write(response.audio_content)
#         print(f'Audio content written to file "{filename}"')  
#     play_mp3(filename)


def start_song(song_name, author):
    # Set your Spotify app credentials
    SPOTIPY_CLIENT_ID = os.getenv("SPOTIPY_CLIENT_ID")
    SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIPY_CLIENT_SECRET")
    SPOTIPY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI")


    # Set the scope to allow control of Spotify playback
    scope = "user-modify-playback-state"

    # Authenticate with Spotify
    sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=SPOTIPY_CLIENT_ID,
                                                   client_secret=SPOTIPY_CLIENT_SECRET,
                                                   redirect_uri=SPOTIPY_REDIRECT_URI,
                                                   scope=scope))

    # Search for the song
    results = sp.search(q=f'track:{song_name} artist:{author}', limit=1)
    if results['tracks']['items']:
        # Get the first song URI
        song_uri = results['tracks']['items'][0]['uri']

        # Start playing the song
        sp.start_playback(uris=[song_uri])
    else:
        print(f"No songs found for {song_name} by {author}")

def get_ssml_from_description(description):
    llm = VertexAI( model_name="text-bison-32k", temperature=temp, max_output_tokens=8100)
    prompt = f"Based on this description, give me the SSML transcription of it using correct emphasis. \nDescription: {description}"
    ouput = llm(prompt)
    return(ouput)

def agent_start(temp: int, address: str, image_path: str):
    llm = ChatVertexAI(model_name="gemini-pro-vision",temperature=temp,max_output_tokens=2040)
    json_format = {
        'description': 'description of the picture.',
        'song_name': 'name of the song',
        'author': 'author of the song',
        'reasoning': 'why this song was selected'
    }
    prompt = f"I am a blind person so I can't see. You are a guide to help me feel the photos I share with you. Describe me this image with many details and be VERY poetic when doing so. Give also details of how the weather should look like. What am I looking at? Based on this image, name me a song I can play to feel this image. The song should reflect the mood of the image and not only describe it. The picture was taken in the following address: {address} \n The response should be in ONLY in json format like this: {json_format} using double quotes."
    

    image_message = {
        "type": "image_url",
        "image_url": {"url": image_path},
    }
    text_message = {
        "type": "text",
        "text": prompt
    }
    message = HumanMessage(content=[text_message, image_message])

    output = llm([message])
    return(output.content)

img = 'image.jpg'
metadata = get_geotagging(img)

# Get the latitude and longitude in decimal degrees
lat = get_decimal_from_dms(metadata['GPSLatitude'], metadata['GPSLatitudeRef'])
lon = get_decimal_from_dms(metadata['GPSLongitude'], metadata['GPSLongitudeRef'])
address = get_location_by_coordinates(lat, lon)
description = agent_start(0, address,img )
description = description.replace('```json', '')
description = description.replace('```', '')
description = json.loads(description)
# ssml = get_ssml_from_description(description["description"])
print(ssml)
print(f"Description: {description['description']}")
print(f"Song Name: {description['song_name']}")
print(f"Author: {description['author']}")
print(f"Reasoning: {description['reasoning']}")
start_song(description['song_name'], description['author'])
text_to_speech(description['description'], 'description.mp3')
text_to_speech(description['reasoning'], 'reasoning.mp3')










