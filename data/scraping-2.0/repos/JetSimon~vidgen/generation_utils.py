from distutils.command.clean import clean
import os
import random
import openai
import pyttsx3
from macos_speech import Synthesizer, AudioFormat
from google_images_search import GoogleImagesSearch
from google.cloud import texttospeech
import storyblocks
import moviepy.editor as mpy
import re
import platform
from keybert import KeyBERT
import pexels
from dotenv import load_dotenv

on_mac = platform.system() == 'Darwin'

print("LOADING ENVIRONMENT VARIABLES")

load_dotenv()

wpm = int(os.getenv('WPM'))
openai.api_key = os.getenv('OPEN_AI_KEY')
pexels.init(os.getenv('PEXELS_KEY'))

google_path = os.getenv("GOOGLE_JSON_PATH")
if google_path:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = google_path
    client = texttospeech.TextToSpeechClient()
else:
    print("GOOGLE_JSON_PATH env variable not set. client will be null!")

GOOGLE_IMAGES_PUBLIC = os.getenv("GOOGLE_IMAGES_PUBLIC")
GOOGLE_IMAGES_PRIVATE = os.getenv("GOOGLE_IMAGES_PRIVATE")

STORYBLOCKS_PUBLIC = os.getenv('STORYBLOCKS_PUBLIC')
STORYBLOCKS_PRIVATE = os.getenv('STORYBLOCKS_PRIVATE')
STORYBLOCKS_PROJECT = os.getenv('STORYBLOCKS_PROJECT')
STORYBLOCKS_USER = os.getenv('STORYBLOCKS_USER')
storyblocks.client(STORYBLOCKS_PUBLIC, STORYBLOCKS_PRIVATE, STORYBLOCKS_PROJECT, STORYBLOCKS_USER)

print("DONE LOADING ENVIRONMENT VARIABLES")

def save_text_and_audio(title, response_text, on_mac = False):
    filename = title_to_filename(title)
    project_path = os.path.join("projects",filename)
    if not os.path.exists(project_path):
        os.makedirs(project_path)
    with open(os.path.join("projects",filename, filename + ".txt"), 'w') as f:
        f.write(response_text)
    if not on_mac:
        audio_path = os.path.join("projects",filename, filename + ".mp3")
        engine = pyttsx3.init()
        engine.setProperty('volume',1.0)
        voices = engine.getProperty('voices') 
        engine.setProperty('voice', voices[0].id)
        engine.setProperty('rate', wpm)
        engine.save_to_file(response_text, audio_path)
        engine.runAndWait()
    else:
        audio_path = os.path.join("projects",filename, filename + ".aiff")
        response_text = response_text.strip().replace("\n"," ")
        format  = AudioFormat('mp3')
        speaker = Synthesizer(voice='Alex', text=response_text, outfile=audio_path, rate=wpm)
        speaker.talk()

def get_google_cloud_voices():
    if google_path:
        return client.list_voices()
    else:
        print("Since GOOGLE_JSON_PATH does not exist, returning empty array for google voices")
        return []

def get_US_google_cloud_voice_names():
    if not google_path:
        print("Since GOOGLE_JSON_PATH does not exist, returning empty array for google voices")
        return []
    output = []
    voices = client.list_voices()
    for voice in voices.voices:
        output.append(voice.name)
    return output

def get_US_google_cloud_voices():
    if not google_path:
        print("Since GOOGLE_JSON_PATH does not exist, returning empty array for google voices")
        return []
    output = []
    voices = client.list_voices()
    for voice in voices.voices:
        if "en-US" in voice.language_codes:
            output.append(voice.name)
    return output

def get_voice_by_name(voice_name):
    if not google_path:
        print("Since GOOGLE_JSON_PATH does not exist, returning None for voice")
        return None
    voices = client.list_voices()
    for voice in voices.voices:
        if voice.name == voice_name:
            return texttospeech.VoiceSelectionParams(
        name=voice_name, language_code=voice.language_codes[0], ssml_gender=texttospeech.SsmlVoiceGender(voice.ssml_gender)
    )
    print("Voice not found with name " + voice_name)
    return None

def save_audio_google_cloud(title, response_text, voice_name="en-US-Wavenet-I", speaking_rate = 1.1, pitch = 0):
    print("Starting Google Cloud TTS")

    script = response_text.replace("\n"," ")
    filename = title_to_filename(title)
    audio_path = os.path.join("projects",filename, filename + ".mp3")
    synthesis_input = texttospeech.SynthesisInput(text=script)
    voice = get_voice_by_name(voice_name)

    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate = speaking_rate, pitch = pitch
    )

    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    with open(audio_path, "wb") as out:
        out.write(response.audio_content)

    print('Finished Google Cloud TTS')

def generate_script(prompt):
    response = openai.Completion.create(
    model="text-davinci-002",
    prompt=prompt,
    temperature=0.7,
    max_tokens=4097-round(len(prompt)/2),
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    return response.choices[0]["text"]

def get_image_from_search(search_term, title, file_name = None):
    folder_name = title_to_filename(title)

    search_params = {
        'q': search_term,
        'num': 10,
        'fileType': 'jpg|png',
        'safe': 'off'
    }

    folder_path = os.path.join("projects",folder_name, "images")

    gis = GoogleImagesSearch(GOOGLE_IMAGES_PUBLIC, GOOGLE_IMAGES_PRIVATE)
    gis.search(search_params=search_params)
    image = random.choice(gis.results())
    image.download(folder_path)  # download image
    _, file_extension = os.path.splitext(image.path)
    os.rename(image.path, os.path.join(folder_path, file_name + file_extension))

# Turns script into a list of trimmed strings separated by ?, ., and !
def clean_script(script):
    response_text = script.strip().replace("\n"," ")
    sentences = re.split(r"\?|\. |\!/g", response_text)
    cleaned_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i].strip()
        if len(sentence) > 0:
            cleaned_sentences.append(sentence)
    return cleaned_sentences

def download_images_from_script(script, title, method = "Pexels"):
    sentences = clean_script(script)

    folder_name = title_to_filename(title)
    folder_path = os.path.join("projects",folder_name, "images")
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print("Extracting keywords")
    extracted = []
    kw_model = KeyBERT()
    index = 0
    for sentence in sentences:
        index += 1
        print(f"Extracting keyword for ({index}/{len(sentences)})")
        keywords = kw_model.extract_keywords(sentence, keyphrase_ngram_range=(1, 2))
        toAdd = sentence
        if len(keywords) > 0:
            toAdd = keywords[0][0]
            print("Extracted " + toAdd)
        else:
            print("Could not find keyword, using sentence")
        extracted.append(toAdd)
    sentences = extracted

    print(f"Downloading media for video using {method}")

    for i in range(len(sentences)):
        print(f"Downloading media for video ({i+1}/{len(sentences)})")
        sentence = sentences[i]

        if method == "Pexels":
            if not get_video_from_pexels(sentence, title, str(i)):
                print(f"Could not find video url on Pexels for {sentence}. Searching google instead.")
                get_image_from_search(sentence, title, str(i))
        elif method == "Storyblocks":
            folder_path = os.path.join("projects", title_to_filename(title), "images")
            if not storyblocks.download_random_video(sentence.replace(" ",", "), os.path.join(folder_path,str(i) + ".mp4")):
                get_image_from_search(sentence, title, str(i))
        else:
            get_image_from_search(sentence, title, str(i))

def get_video_from_pexels(sentence, title, filename):
    folder_name = title_to_filename(title)
    path = os.path.join("projects", folder_name, "images", filename + ".mp4")
    return pexels.download_random_video(sentence.replace(" ",", "), path)

def generate_movie(script, title):
    folder_name = title_to_filename(title)
    sentences = clean_script(script)

def number_of_words(string):
    return len(string.split(" "))

def load_script(path):
    lines = None
    with open(path) as f:
        lines = f.read().replace("\n"," ")
    return clean_script(lines)

def load_script_raw(path):
    lines = None
    with open(path) as f:
        lines = f.read().replace("\n"," ")
    return lines

def title_to_filename(title):
    return title.replace(" ", "").replace("?","").replace("!","")