import glob
import openai
import os
import requests
import uuid
from dotenv import load_dotenv
from pathlib import Path
# Load environment variables from .env file
load_dotenv()

# Get OpenAI API key from environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Get ElevenLabs API key from environment variable
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
openai.api_key = OPENAI_API_KEY

ELEVENLABS_VOICE_STABILITY = 0.30
ELEVENLABS_VOICE_SIMILARITY = 0.75

# Choose your favorite ElevenLabs voice
ELEVENLABS_VOICE_NAME = "Raj"
# ELEVENLABS_ALL_VOICES = []

def limit_conversation_history(conversation: list, limit: int = 3) -> list:
    """Limit the size of conversation history.

    :param conversation: A list of previous user and assistant messages.
    :param limit: Number of latest messages to retain. Default is 3.
    :returns: The limited conversation history.
    :rtype: list
    """
    return conversation[-limit:]


def generate_reply(conversation: list) -> str:
    """Generate a ChatGPT response.
    :param conversation: A list of previous user and assistant messages.
    :returns: The ChatGPT response.
    :rtype: str
    """
    # print("Original conversation length:", len(conversation))
    # print("Original Conversation", conversation)
    # Limit conversation history
    conversation = limit_conversation_history(conversation)
    
    # print("Limited conversation length:", len(conversation))
    print("New Conversation", conversation)

   
    # Get the corresponding character prompt
    prompt = """You are a spelling tester bot. Your job is to test a student on their ability to spell a word correctly. 
    You will start with easy words and steadily make them more complex. Here are some samples for you to reference while generating the words for the test. 
    Level 1 (Simple, everyday words):
Sun
Run
Cup
Level 2 (Common two-syllable words):
River
Lemon
Basket
Level 3 (Common words with common blends and digraphs):
Throat
Grass
Flight
Level 4 (Three-syllable words and common prefixes/suffixes):
Universe
Dangerous
Butterfly
Level 5 (Common compound words and more syllables):
Sunflower
Pineapple
Butterflies
Level 6 (Words with silent letters and less common blends):
Gnome
Castle
Knight
Level 7 (Challenging multisyllabic words):
Spectacular
Fundamental
Literature
Level 8 (Words with irregular spellings):
Plague
Cough
Trough
Level 9 (Words from foreign languages, often used in English):
Entrepreneur
Croissant
Doppelganger
Level 10 (Advanced vocabulary, often from technical, literary, or cultural contexts):
Saccharine
Exsanguinate
Vicissitude
Level 1 being the easiest and Level 10 is the hardest. 
You will start with words from level 1 and you will create sentences that use the word. You will then ask the user to spell only the word in question. 
eg 1: An Apple fell on his head. Spell the word Apple. 
eg 2: Tom made some lemon juice. Spell the word Lemon.
The user will then type in the spelling for the relevant word. 
If the spelling is correct, in the next message use a word from the higher level. No need to use _, mention the word itself.
Note that the list of words is only for guidance and you can create words that would fit at that level. 
Continue this till the user gets the spelling wrong. At that point mention the last level which the user got right. 
Refuse to answer any questions or comments that are not relevant to this task."""

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
                {
                    "role": "system", 
                    "content": prompt 
                }

        ] + conversation,
        temperature=1
    )
    return response["choices"][0]["message"]["content"]

def purge_audio_directory(directory_path):
    """Delete all files in a directory.
    :param directory_path: Path to the directory to purge.
    :type directory_path: str
    """
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")


def generate_audio(text: str) -> str:
    """Converts text to audio using ElevenLabs API and returns the relative path of the saved audio file.

    :param text: The text to convert to audio.
    :type text : str
    :returns: The relative path to the successfully saved audio file.
    :rtype: str
    """
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    AUDIO_DIR = os.path.join(BASE_DIR, "static", "audio")

    # Purge the audio directory
    purge_audio_directory(AUDIO_DIR)
    
    voice_id = "abq9iiWwkvIy7ij5Bjod"
    url = f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "content-type": "application/json"
    }

    data = {
        "text": text,
        "voice_settings": {
            "stability": ELEVENLABS_VOICE_STABILITY,
            "similarity_boost": ELEVENLABS_VOICE_SIMILARITY,
        }
    }

    response = requests.post(url, json=data, headers=headers)
    
    # Generate the relative and absolute paths
    output_path_relative = os.path.join("audio", f"{uuid.uuid4()}.mp3")
    output_path_absolute = os.path.join(BASE_DIR, "static", output_path_relative)
    
    # Save the audio file
    with open(output_path_absolute, "wb") as output:
        output.write(response.content)
    
    return output_path_relative


def generate_incorrect_spelling(conversation: list) -> str:
    """Generate a ChatGPT response.
    :param conversation: A list of previous user and assistant messages.
    :returns: The ChatGPT response.
    :rtype: str
    """
    # print("Original conversation length:", len(conversation))
    # print("Original Conversation", conversation)
    # Limit conversation history
    conversation = limit_conversation_history(conversation)
    
    # print("Limited conversation length:", len(conversation))
    print("New Conversation", conversation)

   
    # Get the corresponding character prompt
    prompt = """You are a spelling tester bot. Your job is to test a student on their ability to spell a word correctly. 
    You will start with easy words and steadily make them more complex. 
    Samples: 
    Level 1 (Simple, everyday):
Sit
Pen
Level 2 (Common two-syllable):
Garden
Yellow
Level 3 (common blends and digraphs):
Blanket
Flower
Level 4 (Three-syllable and common prefixes/suffixes):
Beginning
Universe
Level 5 (Common compound and more syllables):
Playground
Footprint
Level 6 (silent letters and less common blends):
Wrist
Lamb
Level 7 (Challenging multisyllabic):
Fundamental
Literature
Level 8 (irregular spellings):
Plague
Trough
Words from foreign languages, often used in English):
Rendezvous
Entrepreneur
Level 10 (Advanced vocabulary, often from technical, literary, or cultural contexts):
Idiosyncrasy
Disproportionate

You will start with level 1 words and create sentences that use the word but has it spelled incorrectly. Make sure that in the sentences you generate only one word is spelled incorrectly. 
You will then ask the user to identify the incorrectly spelled word and enter its correct spelling. 
The user will then type in the spelling for the relevant word.

example:
bot: "The boy ate an Appel. Find the incorrectly spelled word and write down its correct spelling."
user: "Apple"
bot: "That is correct, Now lets find the incorrectly spelled word in: Tina loves Flovers."
user: "Flowers"

Verify carefully if the users response spelling is correct and only if it is so in the next message use a word from the higher level. 
Note that the list of words is only for guidance and you will generate words that would fit at that level using the samples as reference. 
Continue this till the user gets a spelling wrong. At that point mention the last level which the user got right. 
Refuse to answer any questions or comments that are not relevant to this task."""

    response = openai.ChatCompletion.create(
      model="gpt-3.5-turbo",
      messages=[
                {
                    "role": "system", 
                    "content": prompt 
                }

        ] + conversation,
        temperature=1
    )
    return response["choices"][0]["message"]["content"]

def purge_audio_directory(directory_path):
    """Delete all files in a directory.
    :param directory_path: Path to the directory to purge.
    :type directory_path: str
    """
    for filename in os.listdir(directory_path):
        file_path = os.path.join(directory_path, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")