import os
import tiktoken
import openai
from dotenv import load_dotenv
load_dotenv()

# discord constants 
DISCORD_EXPORT_DIR_PATH = os.getenv('DISCORD_EXPORT_DIR_PATH') 
DISCORD_EXPORT_DIR_PATH_RAW = os.getenv('DISCORD_EXPORT_DIR_PATH_RAW') 
DISCORD_TOKEN_ID = os.getenv('DISCORD_TOKEN_ID')

# html template
HTML_TEMPLATE_PATH_RAW = os.getenv('HTML_TEMPLATE_PATH_RAW')
BULLETINS_PATH_RAW = os.getenv('BULLETINS_PATH_RAW')

# openai constants
COMPLETIONS_MODEL = os.getenv("COMPLETIONS_MODEL")
ENCODING = tiktoken.get_encoding("cl100k_base")

# set my API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# relevant channels
CHANNEL_AND_THREAD_IDS = {
    'lectures': {
        'id': '902967453183778836',
    },
    'staying_ahead_ai_accel': {
        'id': '1088550425524961360'
    },
    'proof_of_building': {
        'id': '1086919421790015629'
    },
    'proof_of_workout': {
        'id': '915652523677868042'
    },
    'educational_contents_for_builders': {
        'id': '1043848628097261658'
    },
    'proof_of_learning': {
        'id': '919209947072434176'
    },
    'building_wealth_and_sharing_alpha': {
        'id': '950184868837490748'
    },
    'tns_nostr': {
        'id': '1073100882700406824'
    },
    # arrow channels
    'arrow_announcements': {
        'id': '909653036887060480'
    },
    'arrow_community_chat': {
        'id': '992121295691063426'
    },
    'arrow_general': {
        'id': '853833144037277729'
    },
    'arrow_ideas': {
        'id': '915000073694371881'
    },
    'arrow_vtol_news': {
        'id': '993543594231222342'
    },
    'arrow_eng_general': {
        'id': '914195759216336947'
    },
    # TODO: i haven't got the rest of the arrow channels, just the ones in "Arrow Community" that seemed relevant as well as an eng chan
    # cabin channels 
    'cabin_general_chat': {
        'id': '892780693652897822'
    }
}

# params for OAI API call
COMPLETIONS_API_PARAMS = {
    "model": COMPLETIONS_MODEL,
    "temperature": 0, # We use temperature of 0.0 because it gives the most predictable, factual answer.
}