import telebot
import requests
import openai
import subprocess
import os
import uuid
import json
import mysql.connector
import logging
from logging.handlers import RotatingFileHandler

# Configura il logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Formattazione per il logger
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Crea un handler di rotazione che scrive nei log con un massimo di 10MB per file e mantiene fino a 5 file di backup.
rotating_handler = RotatingFileHandler('app.log', maxBytes=10*1024*1024, backupCount=5, encoding='utf-8')
rotating_handler.setFormatter(formatter)

# Crea un handler per la console
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

# Aggiungi entrambi gli handler al logger
logger.addHandler(rotating_handler)
logger.addHandler(console_handler)
from cryptography.fernet import Fernet

languages = {
    "English": "en",
    "Italiano": "it",
    "Français": "fr",
    "Español": "es",
    "Deutsch": "de"
}

try:
    with open('languages.json', 'r') as lang_file:
        lang_resources = json.load(lang_file)
        phrases = lang_resources['phrases']
except Exception as e:
    logger.error("Error reading languages.json: %s", e)

with open('config.json', 'r') as config_file:
    config = json.load(config_file)
    cipher = Fernet(config['encryption_key'])
    db_config = config['db_config']

#openai.api_key = config['openai_api_key']
bot_token = config['bot_token'] 
bot = telebot.TeleBot(bot_token)

try:
    bot_info = bot.get_me()
    logger.debug("%s %s", "Bot connected with id: ", bot_info.id)
except Exception as e:
    logger.error("%s %s", "Bot connection error: ", e)

def encrypt_data(data):
    try:
        encrypted_data = cipher.encrypt(data.encode())
        return encrypted_data
    except Exception as e:
        logger.error("Error encrypting data: %s", e)

def decrypt_data(encrypted_data):
    decrypted_data = cipher.decrypt(encrypted_data).decode()
    return decrypted_data

def get_api_key_from_db(chat_id):
    cnx = mysql.connector.connect(**db_config)
    cursor = cnx.cursor()
    cursor.execute("SELECT encrypted_openai_api_key FROM users WHERE chat_id = %s", (chat_id,))
    encrypted_api_key = cursor.fetchone()
    cursor.close()
    cnx.close()

    if encrypted_api_key is None or not encrypted_api_key[0]:
        return None

    decrypted_key = decrypt_data(encrypted_api_key[0])
    return decrypted_key

def store_api_key_in_db(chat_id, api_key):
    encrypted_api_key = encrypt_data(api_key)
    cnx = mysql.connector.connect(**db_config)
    cursor = cnx.cursor()
    cursor.execute("INSERT INTO users (chat_id, encrypted_openai_api_key) VALUES (%s, %s) ON DUPLICATE KEY UPDATE encrypted_openai_api_key = %s", 
                   (chat_id, encrypted_api_key, encrypted_api_key))
    cnx.commit()
    cursor.close()
    cnx.close()

def store_provided_key(message):
    chat_id = message.chat.id
    bot.clear_step_handler_by_chat_id(chat_id=chat_id)  

    api_key = message.text
    if message.text.upper() == "NO":
        bot.clear_step_handler_by_chat_id(chat_id=chat_id)
        bot.send_message(chat_id, operation_cancelled)
        return    
    if is_valid_openai_key(api_key):
        store_api_key_in_db(chat_id, api_key)
        bot.reply_to(message, key_stored)
    else:
        bot.reply_to(message, invalid_api_key)
        bot.register_next_step_handler(message, store_provided_key)  

def is_valid_openai_key(api_key):
    openai.api_key = api_key
    try:
        models = openai.Model.list()
        if models:  
            return True
    except Exception as e:
        logger.error( "%s %s", "Error during OpenAI API call:", e)
        return False 

def remove_phrases(text):
    for phrase in phrases:
        text = text.replace(phrase, "")
    return text

def generate_corrected_transcript(temperature, system_prompt, audio_file):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            temperature=temperature,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": audio_file
                }
            ]
        )
    except openai.error.InvalidRequestError as e:
        logger.debug("%s %s", "Error using GPT-4:", e)
        logger.debug("Trying with GPT-3.5")
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=temperature,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": audio_file
                    }
                ]
            )
        except openai.error.OpenAIError as e:
            logger.debug("%s %s", "Error using GPT-3.5:", e)
            error_message = f"Error using GPT-3.5: {e}"
            return error_message
    except openai.error.OpenAIError as e:
        logger.debug("%s %s", "Error using API:", e)
        error_message = f"Error using API: {e}"
        return error_message        

    return response['choices'][0]['message']['content']

def change_api_key_step(message):
    new_api_key = message.text
    chat_id = message.chat.id
    if message.text == "NO":
        bot.clear_step_handler_by_chat_id(chat_id=chat_id)
        bot.send_message(chat_id, operation_cancelled)
        return    
    if is_valid_openai_key(new_api_key):
        store_api_key_in_db(chat_id, new_api_key)
        bot.send_message(message.chat.id, key_updated)
    else:
        bot.send_message(message.chat.id, invalid_api_key)
        bot.register_next_step_handler(message, change_api_key_step)

def get_language_from_db(chat_id):
    cnx = mysql.connector.connect(**db_config)
    cursor = cnx.cursor()
    cursor.execute("SELECT language FROM users WHERE chat_id = %s", (chat_id,))
    lang = cursor.fetchone()
    cursor.close()
    cnx.close()
    return lang[0] if lang else "en"  

def store_language_in_db(chat_id, lang):
    cnx = mysql.connector.connect(**db_config)
    cursor = cnx.cursor()
    cursor.execute("INSERT INTO users (chat_id, language) VALUES (%s, %s) ON DUPLICATE KEY UPDATE language = %s", 
                   (chat_id, lang, lang))
    cnx.commit()
    cursor.close()
    cnx.close()

def insert_interaction_into_db(chat_id):   
    logger.info ("%s %s", "ChatID: ", chat_id)
    cnx = mysql.connector.connect(**db_config)
    cursor = cnx.cursor()
    cursor.execute("INSERT INTO interactions (ChatID) VALUES (%s)", (chat_id,))
    cnx.commit()
    cursor.close()
    cnx.close()     

def load_language_resources(lang):
    global phrases, system_prompt, welcome_message, help_message, recognition_started, transcription_error
    global provide_api_key, invalid_api_key, key_stored
    global key_updated, operation_cancelled, language_choice, language_error, select_lang, wrong_file_type

    try:
        with open('languages.json', 'r') as lang_file:
            lang_resources = json.load(lang_file)
            phrases = lang_resources['phrases']
            system_prompt = lang_resources[lang]['system_prompt']
            welcome_message = lang_resources[lang]['welcome_message']
            help_message = lang_resources[lang]['help_message']
            recognition_started = lang_resources[lang]['recognition_started']
            transcription_error = lang_resources[lang]['transcription_error']
            provide_api_key = lang_resources[lang]['provide_api_key']
            invalid_api_key = lang_resources[lang]['invalid_api_key']
            key_stored = lang_resources[lang]['key_stored']
            key_updated = lang_resources[lang]['key_updated']
            operation_cancelled= lang_resources[lang]['operation_cancelled']
            language_choice = lang_resources[lang]['language_choice']
            language_error = lang_resources[lang]['language_error']
            select_lang = lang_resources[lang]['select_lang']       
            wrong_file_type = lang_resources[lang]['wrong_file_type']        
    except Exception as e:
        logger.error("Error reading languages.json: %s", e)

def set_language(message):
    chat_id = message.chat.id
    lang_choice = message.text

    if lang_choice in languages:
        language = languages[lang_choice]
        store_language_in_db(chat_id, language)
        load_language_resources(language)
        markup = telebot.types.ReplyKeyboardRemove()
        bot.send_message(chat_id, f"{language_choice} {lang_choice}!", reply_markup=markup)
    else:
        bot.send_message(chat_id, language_error)
        send_language_keyboard(chat_id)

def send_language_keyboard(chat_id):
    markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    for lang in languages:
        markup.add(lang)
    
    bot.send_message(chat_id, select_lang, reply_markup=markup)
    bot.register_next_step_handler_by_chat_id(chat_id, set_language)

# Definizione della funzione split_message
def split_message(message, size=4096):
    words = message.split()
    chunks = []
    current_chunk = words[0]

    for word in words[1:]:
        if len(current_chunk) + len(word) + 1 <= size:
            current_chunk += " " + word
        else:
            chunks.append(current_chunk)
            current_chunk = word

    chunks.append(current_chunk)
    return chunks  

@bot.message_handler(commands=["help"])
def send_help(message):
    chat_id = message.chat.id
    current_language = get_language_from_db(chat_id)
    load_language_resources(current_language)   
    bot.reply_to(
        message, help_message,
    )
@bot.message_handler(commands=["start"])
def send_welcome(message):
    chat_id = message.chat.id
    current_language = get_language_from_db(chat_id)
    load_language_resources(current_language)
    bot.reply_to(message, welcome_message)

@bot.message_handler(commands=["changekey"])
def change_api_key_command(message):
    chat_id = message.chat.id
    current_language = get_language_from_db(chat_id)
    load_language_resources(current_language)
    bot.send_message(message.chat.id, provide_api_key)
    bot.register_next_step_handler(message, change_api_key_step)

@bot.message_handler(commands=["changelanguage"])
def change_language_command(message):
    chat_id = message.chat.id
    current_language = get_language_from_db(chat_id)
    load_language_resources(current_language)
    markup = telebot.types.ReplyKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True)
    for lang_name in languages:
        markup.add(telebot.types.KeyboardButton(lang_name))
    bot.send_message(message.chat.id, select_lang, reply_markup=markup)
    bot.register_next_step_handler(message, set_language)


@bot.message_handler(content_types=['voice', 'audio', 'document'])
def handle_media_messages(message):
    chat_id = message.chat.id
    current_language = get_language_from_db(chat_id)
    load_language_resources(current_language)
    api_key = get_api_key_from_db(chat_id)

    if not is_valid_openai_key(api_key):
        bot.reply_to(message, invalid_api_key)
        bot.register_next_step_handler(message, change_api_key_step)
        return
    
    insert_interaction_into_db(chat_id)
    logger.debug("%s %s", "ChatId: ", chat_id)
    openai.api_key = api_key
    
    # Determina il tipo di file e ottieni il file_path corrispondente
    if message.content_type == 'voice':
        file_info = bot.get_file(message.voice.file_id)
    elif message.content_type == 'audio':
        file_info = bot.get_file(message.audio.file_id)
    elif message.content_type == 'document':
        # Verifica se il documento è un file audio per estensione o MIME type
        if message.document.mime_type.startswith('audio') or message.document.file_name.split('.')[-1] in ['mp3', 'wav', 'ogg']:
            file_info = bot.get_file(message.document.file_id)
        else:
            bot.reply_to(message, wrong_file_type)
            return
    else:
        # Questo caso non dovrebbe verificarsi, ma è un'ulteriore misura di sicurezza
        bot.reply_to(message, "Tipo di file non supportato.")
        return

    file_path = file_info.file_path
    file_url = f'https://api.telegram.org/file/bot{bot_token}/{file_path}'

    logger.info('Downloading audio file...')
    audio_file = requests.get(file_url)

    file_extension = file_path.split('.')[-1]
    file_name = f'{str(uuid.uuid4())}.{file_extension}'
    with open(file_name, 'wb') as f:
        f.write(audio_file.content)

    sent_message = bot.reply_to(message, recognition_started, parse_mode='Markdown')
    logger.info('Converting audio file in MP3 format...')
    mp3_file_name = f'{str(uuid.uuid4())}.mp3'
    try:
        subprocess.run(['ffmpeg', '-i', file_name, mp3_file_name])
    except Exception as e:
        logger.error("Error converting audio file with FFmpeg: %s", e)    

    transcript = None
    try:
        with open(mp3_file_name, "rb") as audio_file:
            transcript = openai.Audio.transcribe(
                file=audio_file,
                model="whisper-1",
                response_format="text"
            )
            logger.debug(transcript)
            transcript = remove_phrases(transcript)
            logger.debug(transcript)
            if transcript is not None and transcript.strip():
                corrected_text = generate_corrected_transcript(0, system_prompt, transcript)
                logger.debug(corrected_text)
                message_parts = split_message(corrected_text)                
                bot.edit_message_text(chat_id=message.chat.id, message_id=sent_message.message_id, text=message_parts[0])
                for part in message_parts[1:]:
                    bot.send_message(chat_id=message.chat.id, text=part)                
            else:
                bot.edit_message_text(chat_id=message.chat.id, message_id=sent_message.message_id, text=transcription_error)
    except Exception as e:
        logger.error("%s %s", "Error transcribing audio file:", e)
        transcription_error_message = f"{transcription_error} {e}"
        bot.edit_message_text(chat_id=message.chat.id, message_id=sent_message.message_id, text=transcription_error_message)

    logger.info('Deleting audio file...')
    try:
        os.remove(file_name)
        os.remove(mp3_file_name)
    except OSError as e:
        logger.error("Error removing file: %s", e)

    logger.info("%s %s", 'Original file name:', file_name)
    logger.info("%s %s", 'Converted file name:', mp3_file_name)

bot.polling()
