'''
    assets_gen.py

    - This script generates the assets for the video (script, audio, images, videos)

    Author: Juled Zaganjori    
'''

import os
import openai
import json
import random
import requests
from dotenv import load_dotenv
import azure.cognitiveservices.speech as speechsdk
import librosa
from tqdm import tqdm

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

# Set up TTS client
speech_key = os.getenv("AZURE_SPEECH_KEY")
service_region = os.getenv("AZURE_SPEECH_REGION")
voices = ["en-US-JennyNeural", "en-US-GuyNeural", "en-US-AriaNeural", "en-US-DavisNeural", "en-US-AmberNeural", "en-US-AnaNeural", "en-US-AshleyNeural", "en-US-BrandonNeural", "en-US-ChristopherNeural", "en-US-CoraNeural", "en-US-ElizabethNeural", "en-US-EricNeural", "en-US-JacobNeural", "en-US-JaneNeural",
          "en-US-JasonNeural", "en-US-MichelleNeural", "en-US-MonicaNeural", "en-US-NancyNeural", "en-US-RogerNeural", "en-US-SaraNeural", "en-US-SteffanNeural", "en-US-TonyNeural", "en-US-AIGenerate1Neural1", "en-US-AIGenerate2Neural1", "en-US-BlueNeural1", "en-US-JennyMultilingualV2Neural1", "en-US-RyanMultilingualNeural1"]
speech_config = speechsdk.SpeechConfig(
    subscription=speech_key, region=service_region)
speech_config.speech_synthesis_voice_name = random.choice(voices)
speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

# Global variables
min_stock_video_length = 5  # seconds
min_stock_image_length = 3  # seconds
max_stock_video_length = 10  # seconds
max_stock_image_length = 5  # seconds
max_paragraphs = 3
orientation = "landscape"
asset_size = "medium"

# Generate random string


def get_random_string(length):
    letters = "abcdefghijklmnopqrstuvwxyz1234567890"
    result_str = "".join(random.choice(letters) for i in range(length))
    return result_str


# Setup video directory
def video_setup():
    global max_paragraphs
    # Generate video ID
    video_id = get_random_string(15)
    if not os.path.exists("videos"):
        os.makedirs("videos")
    # Save output
    if not os.path.exists("videos/" + video_id):
        os.makedirs("videos/" + video_id)
    else:
        video_id = get_random_string(20)

    for i in range(0, max_paragraphs):
        os.makedirs("videos/" + video_id + "/p" + str(i) + "/img")

    for i in range(0, max_paragraphs):
        os.makedirs("videos/" + video_id + "/p" + str(i) + "/video")

    return video_id


# Video script from OpenAI
def get_video_script(topic, video_id):
    global max_paragraphs

    # Prompt
    prompt = '''
        You are a video script generation machine. I give you a topic and you create 3 paragraphs of video script with an intro and an outro. You should output only in JSON format and separate each paragraph in with a different key "P1", "P2", "P3".  You should also include strings in [] where you should include tags for an image that you find reasonable to display in that moment in time. There should be 10 tags minimum in each such as ["black coat", "dressing room", "wardrobe", "HD", "man"... ]. Make sure to include a variety of these tags in different points in time so that the article images correspond and are abundant.
        Please stick to the format. Paragraphs are only text and tags are only strings in []. You can't use special characters. DON'T ADD ANYTHING ELSE TO THE RESPONSE. ONLY THE JSON FORMAT BELOW.
        Here's the format of what I'm looking for (NEVER GO OUT OF THIS FORMAT AND CHANGE THE DICTIONARY KEYS):
        {
            "topic": " '''+topic+''' ",
    '''

    # Create a prompt sample as the one above but as many max_paragraphs value
    for i in range(0, max_paragraphs):
        prompt += '''
                "p''' + str(i) + '''": "paragraph text",
                "p''' + str(i) + '''_img_tags": [...],
        '''

    prompt += '''
        }
    '''

    # Completion
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": prompt
            },
            {
                "role": "user",
                "content": topic
            }
        ],
    )

    # Validate output
    try:
        # Sanitize output
        transcript = sanitize_JSON(
            response["choices"][0]["message"]["content"])

        json.loads(transcript)
        with open("videos/" + video_id + "/script.json", "w") as f:
            f.write(transcript)

        return True

    except Exception as e:
        print(e)
        return False


def sanitize_JSON(json_string):
    json_string = str(json_string)
    # Remove \n
    json_string = json_string.replace("\n", "")
    # Remove \"
    json_string = json_string.replace("\\\"", "\"")
    # Remove \'
    json_string = json_string.replace("\\\'", "\'")
    # Remove \\n
    json_string = json_string.replace("\\n", "")

    return json_string

# TTS audio from Google Cloud


def get_tts_audio(video_id):
    global max_paragraphs
    # Read script
    with open("videos/" + video_id + "/script.json", "r") as f:
        script = json.loads(f.read())

    # for i in range(0, max_paragraphs):
    for i in tqdm(range(0, max_paragraphs)):
        # Generate audio
        result = speech_synthesizer.speak_text_async(
            script["p" + str(i)]).get()
        # Check result
        if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
            # Saves the synthesized speech to an audio file.
            stream = speechsdk.AudioDataStream(result)
            stream.save_to_wav_file(
                "videos/" + video_id + "/p" + str(i) + "/audio.wav")

    return True


# Photo and video assets
def get_stock_images(video_id, part_number, part_tags, image_count, orientation, asset_size):
    api_key = os.getenv("PEXELS_API_KEY")
    # Perform search with the tags joined by a + sign
    response = requests.get("https://api.pexels.com/v1/search?query=" + "+".join(part_tags) + "&per_page=" +
                            str(image_count) + "&orientation=" +
                            orientation + "&size=" + str(asset_size),
                            headers={"Authorization": api_key})
    # Get images
    images = response.json()["photos"]
    # Get image URLs
    image_urls = [image["src"]["original"] for image in images]
    # Download images
    for i in range(0, len(image_urls)):
        # Get image
        image = requests.get(image_urls[i])
        # Save image
        with open("videos/" + video_id + "/p" + str(part_number) + "/img/" + str(i) + ".jpg", "wb") as f:
            f.write(image.content)


def get_stock_videos(video_id, part_number, part_tags, video_count, orientation, asset_size):

    api_key = os.getenv("PEXELS_API_KEY")

    response = requests.get("https://api.pexels.com/videos/search?query=" + "+".join(
        part_tags) + "&orientation=" + orientation + "&size=" + str(asset_size) + "&per_page=" + str(video_count),
        headers={"Authorization": api_key})
    # Get videos
    videos = response.json()["videos"]

    # Get video URLs
    video_urls = [video["video_files"][0]["link"] for video in videos]

    # Download videos
    for i in range(0, video_count):
        # Get video
        video = requests.get(video_urls[i])
        # Save video
        with open("videos/" + video_id + "/p" + str(part_number) + "/video/" + str(i) + ".mp4", "wb") as f:
            f.write(video.content)


# Setup stock assets
def get_part_stock_assets(video_id, part_num, part_len):
    global orientation, asset_size

    # Read tags from script.json
    with open("videos/" + video_id + "/script.json", "r") as f:
        script = json.loads(f.read())

    # Get tags
    part_tags = script["p" + str(part_num) + "_img_tags"]

    img_count = int(part_len / min_stock_image_length / 2)
    video_count = int(part_len / min_stock_video_length / 2)

    get_stock_images(video_id, part_num, part_tags,
                     img_count, orientation, asset_size)
    get_stock_videos(video_id, part_num, part_tags,
                     video_count, orientation, asset_size)


def get_stock_assets(video_id):
    global max_paragraphs
    # Read script.json
    with open("videos/" + video_id + "/script.json", "r") as f:
        script = json.loads(f.read())

    # Calculate part lengths from the audios
    part_lengths = []
    for i in range(0, max_paragraphs):
        # Get audio length
        audio_length = librosa.get_duration(path="videos/" + video_id + "/p" + str(i) + "/audio.wav")
        part_lengths.append(audio_length)

    print("Downloading assets...")
    # Get stock assets for each part
    for i in tqdm(range(0, len(part_lengths))):
        get_part_stock_assets(video_id, i, part_lengths[i])

    return True


def assets_gen(topic, custom_orientation="landscape", custom_asset_size="medium"):
    global orientation, asset_size
    orientation = custom_orientation
    asset_size = custom_asset_size

    # Setup video
    video_id = video_setup()
    # Get video script
    print("Generating video script...")
    if get_video_script(topic, video_id):
        print("Video script generated!")
    else:
        print("Video script generation failed!")
    # Get TTS audio
    print("Generating TTS audio...")
    if get_tts_audio(video_id):
        print("TTS audio generated!")
    else:
        print("TTS audio generation failed!")
    # Get stock assets
    print("Generating stock assets...")
    if get_stock_assets(video_id):
        print("Stock assets generated!")
    else:
        print("Stock assets generation failed!")

    return video_id


if __name__ == "__main__":
    # Get topic
    topic = input("Enter a topic: ")
    # Setup video
    video_id = video_setup()
    # Get video script
    print("Generating video script...")
    if get_video_script(topic, video_id):
        print("Video script generated!")
    else:
        print("Video script generation failed!")
    # Get TTS audio
    print("Generating TTS audio...")
    if get_tts_audio(video_id):
        print("TTS audio generated!")
    else:
        print("TTS audio generation failed!")
    # Get stock assets
    print("Generating stock assets...")
    if get_stock_assets(video_id):
        print("Stock assets generated!")
    else:
        print("Stock assets generation failed!")
