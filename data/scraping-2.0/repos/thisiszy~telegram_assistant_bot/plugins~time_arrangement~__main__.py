from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes, CommandHandler, filters, ConversationHandler, MessageHandler, CallbackQueryHandler
import logging
import configparser
from datetime import datetime
import json
from tzlocal import get_localzone
import re
# import openai GPT-3 token
import openai
config = configparser.ConfigParser()
config.read('config.ini')
token = config['OPENAI']['ACCESS_TOKEN_GPT3']
openai.api_key = token

# import whisper
import whisper
import shutil
import os
AUDIO_FILE_PATH = "env/share/whisper/audio"

if os.path.exists(AUDIO_FILE_PATH):
    shutil.rmtree(AUDIO_FILE_PATH)
os.makedirs(AUDIO_FILE_PATH)

# use reverse chatgpt
from revChatGPT.V3 import Chatbot

# use google calendar
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from google.auth import exceptions
# If modifying these scopes, delete the file token.json.
SCOPES = ['https://www.googleapis.com/auth/calendar']

# Set up the SQLite database
import sqlite3
conn = sqlite3.connect('storage.db')
c = conn.cursor()
c.execute('''
    CREATE TABLE IF NOT EXISTS tokens
    (user_id INTEGER PRIMARY KEY, token TEXT, credentials TEXT)
''')

chatbot = Chatbot(api_key=config['OPENAI']['API_KEY'], engine='gpt-3.5-turbo')

BASE_PROMPT = 'Extract the activity or event name, place, start time, end time in the format "{{"name":  "", "place": "", "stime": "", "etime": ""}}" from the following sentence: "{0}". The output should obey the following rules: 1. If any of the item is empty, use "None" to replace it. 2. name, "start time" and "end time" is mandatory. 3. "start time" and "end time" should be represented by "yyyy-mm-dd hh:mm:ss" in 24-hour clock format. Current time is {1}, it\'s {2}. 4. If there is no end time, you should assume the end time is one hour later than the start time. 5. If there are multiple different results, you should list them in different lines 6. Your response should not contain anything unrelated to the format above.'

def get_info():
    return {
        "name": "time_arrangement", 
        "version": "1.0.0", 
        "author": "thisiszy",
        "description": "*time\_arrangement*: arrange time by text and voice, use /schedule to start, use /stopschedule to stop",
        "commands": [""],
        "message_type": ["text", "audio"]
    }

WAITING, ADDING, SELECTING = range(3)

def get_handlers(command_list):
    info = get_info()
    handlers = [ConversationHandler(
        entry_points=[CommandHandler("schedule", start)],
        states={
            WAITING: [MessageHandler(filters.TEXT & (~filters.COMMAND) | filters.VOICE, arrange_time_chatgpt)],
            SELECTING: [CallbackQueryHandler(selecting_calendar_callback)],
            ADDING: [CallbackQueryHandler(modify_calendar_callback)],
        },
        fallbacks=[CommandHandler("stopschedule", cancel)],
    )]
    logging.log(logging.INFO, f"Loaded plugin {info['name']}, commands: {info['commands']}, message_type: {info['message_type']}")
    return handlers, info

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Send me text or audio, I can arrange time for you.\n\n"
    )
    return WAITING

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Cancels and ends the conversation."""
    await update.message.reply_text(
        "Canceled."
    )
    return ConversationHandler.END

async def arrange_time_gpt3(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = None
    place_holder = None
    try:
        if update.message.text is not None:
            prompt = BASE_PROMPT.format(update.message.text, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), datetime.weekday(datetime.now()))
        if update.message.voice is not None:
            place_holder = await context.bot.send_message(chat_id=update.effective_chat.id, text="Converting...", reply_to_message_id=update.message.message_id)
            file = await update.message.voice.get_file()
            audio_file_path = os.path.join(AUDIO_FILE_PATH, file.file_id)
            await file.download_to_drive(audio_file_path)
            logging.log(logging.INFO, f"Received audio file: {audio_file_path}")
            model = whisper.load_model("small", download_root="env/share/whisper")
            result = model.transcribe(audio_file_path)
            logging.log(logging.INFO, f"Recognized text: {result['text']}")
            prompt = BASE_PROMPT.format(result["text"], datetime.now().strftime("%Y-%m-%d %H:%M:%S"), datetime.weekday(datetime.now()))
        if prompt is not None:
            if place_holder is not None:
                await context.bot.edit_message_text(
                    chat_id=place_holder.chat_id, message_id=place_holder.message_id, text="Parsing..."
                )
            else:
                place_holder = await context.bot.send_message(chat_id=update.effective_chat.id, text="Parsing...", reply_to_message_id=update.message.message_id)
            gpt_model = "text-davinci-003"
            temperature = 0.5
            max_tokens = 4000

            # Generate a response
            response = openai.Completion.create(
                engine=gpt_model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            response = response.choices[0].text.strip()
            logging.info(f"Response: {response}")
            response = response.replace("\n", "")
            pattern = re.compile(r'{ *"name":.*, *"place":.*, *"stime": *"\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}" *, *"etime": *"\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}" *}', flags=0)
            matched = pattern.findall(response)
            if len(matched) > 0:
                keyboard = [
                    [
                        InlineKeyboardButton("Apply", callback_data="Y"),
                        InlineKeyboardButton("Cancel", callback_data="N"),
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                context.user_data["message"] = (update.message.text, matched[-1], place_holder.message_id)
                await context.bot.edit_message_text(
                    chat_id=place_holder.chat_id,
                    message_id=place_holder.message_id,
                    text=matched[-1],
                    reply_markup=reply_markup
                )
                return ADDING
            else:
                await context.bot.edit_message_text(
                    chat_id=place_holder.chat_id, message_id=place_holder.message_id, text=f"Not matched: {response}\nExit conversation"
                )
                return ConversationHandler.END
    except Exception as e:
        logging.log(logging.ERROR, f"Error: {e}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text="Error f{e}\nExit conversation", reply_to_message_id=update.message.message_id)
        return ConversationHandler.END

async def arrange_time_chatgpt(update: Update, context: ContextTypes.DEFAULT_TYPE):
    prompt = None
    place_holder = None
    response = ""
    try:
        if update.message.text is not None:
            prompt = BASE_PROMPT.format(update.message.text, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), datetime.weekday(datetime.now()))
        if update.message.voice is not None:
            place_holder = await context.bot.send_message(chat_id=update.effective_chat.id, text="Converting...", reply_to_message_id=update.message.message_id)
            file = await update.message.voice.get_file()
            audio_file_path = os.path.join(AUDIO_FILE_PATH, file.file_id)
            await file.download_to_drive(audio_file_path)
            model = whisper.load_model("small", download_root="env/share/whisper")
            result = model.transcribe(audio_file_path)
            logging.debug(f"Received audio file: {audio_file_path}")
            logging.debug(f"Recognized text: {result['text']}")
            prompt = BASE_PROMPT.format(result["text"], datetime.now().strftime("%Y-%m-%d %H:%M:%S"), datetime.weekday(datetime.now()))
        if prompt is not None:
            logging.debug(f"Prompt: {prompt}")
            if place_holder is not None:
                await context.bot.edit_message_text(
                    chat_id=place_holder.chat_id, message_id=place_holder.message_id, text="Parsing..."
                )
            else:
                place_holder = await context.bot.send_message(chat_id=update.effective_chat.id, text="Parsing...", reply_to_message_id=update.message.message_id)

            logging.info(f"Prompt: {prompt}")
            response = chatbot.ask(prompt)
            logging.info(f"Response: {response}")
            # response = response.replace("\n", "")
            pattern = re.compile(r'{[\n\t ]*"name":.*,[\n\t ]*"place":.*,[\n\t ]*"stime":[\n\t ]*"\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}"[\n\t ]*,[\n\t ]*"etime":[\n\t ]*"\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}"[\n\t ]*}', flags=0)
            matched = pattern.findall(response)
            if len(matched) > 0:
                events_json = []
                events_text = []
                keyboard = [[]]
                for idx, item in enumerate(matched):
                    events_json.append(json.loads(item))
                    events_text.append(f"{idx}. {json.dumps(events_json[-1])}")
                    keyboard[0].append(InlineKeyboardButton(f"{idx}", callback_data=f"{idx}"))
                keyboard.append([InlineKeyboardButton("Cancel", callback_data="N")])
                # keyboard = [
                #     [
                #         InlineKeyboardButton("Apply", callback_data="Y"),
                #         InlineKeyboardButton("Cancel", callback_data="N"),
                #     ]
                # ]
                context.user_data["message"] = (update.message.text, events_json, place_holder)
                reply_markup = InlineKeyboardMarkup(keyboard)
                await context.bot.edit_message_text(
                    chat_id=place_holder.chat_id,
                    message_id=place_holder.message_id,
                    text="\n".join(events_text),
                )
                await context.bot.edit_message_reply_markup(
                    chat_id=place_holder.chat_id,
                    message_id=place_holder.message_id,
                    reply_markup=reply_markup
                )
                
                return SELECTING
            else:
                await context.bot.edit_message_text(
                    chat_id=place_holder.chat_id, message_id=place_holder.message_id, text=f"Not matched: {response}\nExit conversation"
                )
                return ConversationHandler.END
    except Exception as e:
        logging.log(logging.ERROR, f"Error: {e}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Error: {e}\nResponse: {response}\nExit conversation", reply_to_message_id=update.message.message_id)

        return ConversationHandler.END

# input: selected event json
# output: ask user to select calendar
async def selecting_calendar_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    orig_text, events, place_holder = context.user_data["message"]
    try:
        if update.callback_query.data != "N":
            calendar_serv = get_calendar_service(update.effective_chat.id)
            calender_list = list_calendar(calendar_serv)

            keyboard = [[]]
            for idx, item in enumerate(calender_list):
                keyboard[0].append(InlineKeyboardButton(f"{item['summary']}", callback_data=f"{idx}"))
            keyboard.append([InlineKeyboardButton("Cancel", callback_data="N")])

            context.user_data["message"] = (orig_text, events[int(update.callback_query.data)], calender_list, place_holder.message_id)
            reply_markup = InlineKeyboardMarkup(keyboard)
            await context.bot.edit_message_reply_markup(
                chat_id=place_holder.chat_id,
                message_id=place_holder.message_id,
                reply_markup=reply_markup
            )

            return ADDING
        else:
            await context.bot.edit_message_text(chat_id=update.effective_chat.id, text=f"Canceled\nschedule exit", message_id=place_holder.message_id)
            return ConversationHandler.END
    except Exception as e:
        logging.log(logging.ERROR, f"Error: {e}")
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"Error: {e}\nExit conversation", reply_to_message_id=update.message.message_id)
        return ConversationHandler.END

# input: selected calendar
# output: add event to calendar
async def modify_calendar_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    orig_text, event, calender_list, message_id = context.user_data["message"]
    try:
        if update.callback_query.data != "N":
            calendar_serv = get_calendar_service(update.effective_chat.id)
            event_id = modify_calendar(orig_text, event, calender_list[int(update.callback_query.data)]['id'], calendar_serv)
            await context.bot.edit_message_text(chat_id=update.effective_chat.id, text=f"{event_id}", message_id=message_id)
            # await context.bot.edit_message_text(chat_id=update.effective_chat.id, text=f"Event added: {event}\nschedule exit", message_id=message_id)
        else:
            await context.bot.edit_message_text(chat_id=update.effective_chat.id, text=f"Canceled\nschedule exit", message_id=message_id)
    except Exception as e:
        logging.log(logging.ERROR, f"Error: {e}")
        await context.bot.edit_message_text(chat_id=update.effective_chat.id, text=f"Error: {e}, Event: {event}\nschedule exit", message_id=message_id)
    finally:
        return ConversationHandler.END

def update_token_crediential(user_id, secret, force_update=False):
    creds = None
    result = None
    # The database stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if not force_update:
        c.execute('SELECT token, credentials FROM tokens WHERE user_id = ?', (user_id,))
        result = c.fetchone()
        secret = result[0]
        if result:
            creds = Credentials.from_authorized_user_info(json.loads(result[1]), SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if secret is not None:
                flow = InstalledAppFlow.from_client_config(json.loads(secret), SCOPES)
            else:
                if result:
                    secret = result[0]
                    if secret is not None:
                        flow = InstalledAppFlow.from_client_config(json.loads(secret), SCOPES)
                    else:
                        logging.error("result[0] is None")
                        return None
                else:
                    logging.error("No secret found")
                    return None
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        if secret is None:
            logging.error("No secret found, can't save")
            logging.error(creds.to_json())
            return None
        c.execute('INSERT OR REPLACE INTO tokens (user_id, token, credentials) VALUES (?, ?, ?)', (user_id, secret, creds.to_json()))
        conn.commit()
    return creds

def get_calendar_service(user_id):
    try:
        creds = update_token_crediential(user_id, None)
    except exceptions.RefreshError:
        c.execute('SELECT token FROM tokens WHERE user_id = ?', (user_id,))
        result = c.fetchone()
        if result:
            creds = update_token_crediential(user_id, result[0], force_update=True)
        else:
            raise exceptions.RefreshError

    service = build('calendar', 'v3', credentials=creds)

    return service

def modify_calendar(orig_text, event, calender_id, service):
    try:
        # get the local time zone
        timezone = datetime.now().astimezone().tzinfo
        timezone_str = str(get_localzone())

        # get the current UTC offset for the local time zone
        utc_offset = timezone.utcoffset(datetime.now()).total_seconds() / 3600

        # format the offset as a string
        offset_str = "{:+03d}:00".format(int(utc_offset))
        # format the datetime object as a string in the desired format
        def format_datetime(time : str) -> str:
            dt = datetime.strptime(time, "%Y-%m-%d %H:%M:%S")
            return dt.strftime("%Y-%m-%dT%H:%M:%S") + offset_str

        event = {
            'summary': event['name'],
            'location': event['place'],
            'description': orig_text,
            'start': {
                'dateTime': format_datetime(event["stime"]),
                'timeZone': timezone_str,
            },
            'end': {
                'dateTime': format_datetime(event["etime"]),
                'timeZone': timezone_str,
            },
            'reminders': {
                'useDefault': False,
                'overrides': [
                    {'method': 'email', 'minutes': 4 * 60},
                    {'method': 'popup', 'minutes': 40},
                ],
            },
        }

        event = service.events().insert(calendarId=calender_id, body=event).execute()
        return event.get('id')


    except HttpError as error:
        return f'An error occurred: {error}'

def list_calendar(service):
    page_token = None
    calendar_list = []
    while True:
        cur_calendar_list = service.calendarList().list(pageToken=page_token).execute()
        for calendar_list_entry in cur_calendar_list['items']:
            calendar_list.append({
                'summary': calendar_list_entry['summary'],
                'id': calendar_list_entry['id']
            })
        page_token = cur_calendar_list.get('nextPageToken')
        if not page_token:
            break
    return calendar_list