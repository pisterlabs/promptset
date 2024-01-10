import json
from dotenv import load_dotenv
import os
import shutil
from pymongo import MongoClient 
from motor.motor_asyncio import AsyncIOMotorClient
import json
import random
import openai
load_dotenv()
import logging


def setup_logging():
    logger = logging.getLogger(__name__)
    
    # Prüfen, ob der Logger bereits Handler hat, um doppelte Initialisierung zu vermeiden
    if not logger.handlers:
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(lineno)s: %(message)s', '%d.%m.%y %H:%M:%S')
        
        # Datei-Handler für die Datei 'actions/exceptions.log'
        datei_handler = logging.FileHandler('actions/exceptions.log', 'a')
        datei_handler.setFormatter(formatter)
        logger.addHandler(datei_handler)

        # Konsole-Handler für die Konsolenausgabe
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.ERROR)
        logger.addHandler(console_handler)

    return logger
    
logger = setup_logging()

def get_dp_inmemory_db(json_file):
    """ load the inmemory db from json file """
    with open(json_file, "r") as jsonFile:
        return json.load(jsonFile)

def async_connect_to_db(database, collection):
    """
    Returns the collections(with the name collection) in the cluster(database)
    """
    try:
        cluster = AsyncIOMotorClient(get_credentials('MONGO_DB_LOCAL'),connectTimeoutMS=120000, serverSelectionTimeoutMS=120000)
        db = cluster[database]
        collections = db[collection]
        return collections
    except Exception as e: 
        logger.exception(e)

def print_current_tracker_state(tracker):
    '''
    Debug purpose: get the current state of the tracker 
    '''
    current_state = tracker.current_state() 
    # Iterate over the keys of the dictionary
    for state in current_state:
        print(state, current_state[state])


def get_groupuser_id_and_answer(tracker):
    '''
    Get Answer and UserID of the gorupmember who is answering the question through custom connector
    '''
    try:
        for event in reversed(tracker.events):
                    if event['event'] == 'user':
                        #print("EVENT",event )
                        if 'groupuser_id' in event['metadata']:
                            groupuser_id = event['metadata']['groupuser_id']
                            groupuser_name = event['metadata']['groupuser_name']

                            if len(event['parse_data']['entities']) > 0:
                                answer = event['parse_data']['entities'][0]['value']
                            else:
                                answer = event['text']
                            return groupuser_id,groupuser_name,answer
                        else: 
                            return None, None, None
    except Exception as e: 
        logger.exception(e)

def get_random_person(group):
    return random.choice(group['users'])

def get_requested_slot(tracker):
    '''
    get current requested slot of form
    '''
    try: 
        current_state = tracker.current_state() 
        for state in reversed(current_state):
            if state == "slots":
                return current_state[state]['requested_slot']
    except Exception as e:
        logger.exception(e)
        return None

def get_credentials(keyname):
    '''
    get value from env file
    '''
    try:
        return os.environ[keyname] if keyname in os.environ else os.getenv(keyname)
    except:
        return os.getenv(keyname)


def ask_openai(role, question, retries=10):
    try: 
        openai.api_key = get_credentials("OPEN_AI")
        completion = openai.ChatCompletion.create(
        model= get_credentials("OPEN_AI_MODEL"),
        messages=[
            {"role": "assistant", "content": "%s %s"%(role,question)}
        ],
        temperature=1,
        max_tokens=256,
        request_timeout=30,
        n =1
        )
        return completion.choices[0].message.content
    except Exception as e:
        logger.exception(e)
        logger.info("OpenAI Read Timed Out!")
        if retries > 0:
            return ask_openai(role, question, retries - 1)
        else: 
            return "🤯 Meine Gedanken sind wie Popcorn, ständig platzen neue Fragen auf - ich kann gerade nicht 🙊 antworten! 😂🍿"


def get_json_msg(recipient_id, text):
    return {
            "chat_id": recipient_id, 
            "text": text,
            "parse_mode": "MarkdownV2",
        }


async def ben_is_typing(countdown, game_mode_handler):
    try:
        await game_mode_handler.telegram_bot_send_message('edit', countdown['sender_id'],"Ben tippt ...", message_id=countdown['message_id'] )
        await game_mode_handler.telegram_bot_send_message('pin', countdown['sender_id'], " ", message_id=countdown['message_id'])
    except Exception as e:
        logger.exception(e)


async def ben_is_typing_2(countdown, game_mode_handler):
    try:
        await game_mode_handler.telegram_bot_send_message('edit', countdown['sender_id'],"Ben tippt ....", message_id=countdown['message_id'] )
        await game_mode_handler.telegram_bot_send_message('pin', countdown['sender_id'], " ", message_id=countdown['message_id'])
    except Exception as e:
        logger.exception(e)

def get_countdown_value(quest_id, loop):
    try: 
        mode = '_'.join(loop.split('_')[2:]) if loop else None
        values = get_dp_inmemory_db("./countdown_values.json")

        # Check if mode and quest_id are valid keys in the dictionaries
        if mode is not None and quest_id is not None and mode in values and quest_id in values.get(mode, {}):
            return values[mode][quest_id]
        else: 
            return 0
    except Exception as e:
        logger.exception(e)
        logger.info("ACTIVE_LOOP: " + loop)
        return 0

def remove_prefix(input_string, prefix):
    try:
        if input_string.startswith(prefix):
            return input_string[len(prefix+"_"):]
        else:
            return input_string
    except Exception as e:
        logger.exception(e)
        return input_string

        
def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

def delete_folder(folder_path):
    try:
        shutil.rmtree(folder_path)
    except OSError as e:
        print(f"Fehler beim rekursiven Löschen des Ordners: {e}")




