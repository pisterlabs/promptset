import os
import openai
import sqlite3
import os
from elevenlabs import generate, set_api_key, save

# Retrieve the API Key from environment variables
api_key = os.getenv('dict_openai_api_secret')

if api_key is None:
    raise ValueError("API_KEY is not set in the environment variables")

# Set up OpenAI with the API Key
openai.api_key = api_key


# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('dictionary_entries.db')
# Create a new SQLite cursor
cur = conn.cursor()

# Retrieve data from the database
cur.execute('SELECT pronunciation,word FROM dictionary_entries ORDER BY word ASC')
rows = cur.fetchall()



def gen_sound(word,pronounciation):
    file=os.path.join('images',word+".mp3")
    api_key=os.environ.get('elevenlabs_api_key')
    set_api_key(api_key)

    resp_audio_file=os.path.join('sounds',word+".mp3")

    
    audio = generate(
    text="The word is pronounced: "+word,
    voice="chris1",
    model="eleven_monolingual_v1"
    )
    save(audio,resp_audio_file)
 

for row in rows:

    gen_sound(row[1],row[0])
