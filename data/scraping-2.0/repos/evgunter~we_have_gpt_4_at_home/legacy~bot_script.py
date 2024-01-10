import os
import dotenv
import logging
import random
import string
from telegram.constants import ParseMode
import telegram.ext
import time
import typing
import openai


# === Setup ============================================================================================================

MODEL = "gpt-4"  # "gpt-3.5-turbo"  # "text-davinci-003"
LONG_MODEL = "gpt-4-32k"
FAST_MODEL = "gpt-3.5-turbo"
LONG_FAST_MODEL = "gpt-3.5-turbo-16k"

SYSTEM_PROMPT = {"role": "system", "content": "You are a helpful Telegram chatbot. You can use Telegram markdown message formatting, e.g. `inline code`, ```c++\ncode written in c++```, *bold*, and _italic_."}

TURBO_COMMAND = 'turbo'
NO_RESPONSE_COMMAND = 'no_response'

class Message:
    def __init__(self, message_id, message_text):
        self.message_id = message_id
        self.message_text = message_text

# https://github.com/python-telegram-bot/python-telegram-bot/wiki/Extensions-%E2%80%93-Your-first-Bot

logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f'gpt_4_telegram_{time.time()}.log'),
    ]
)

conversations = {}  # current conversations for each chat

# === Telegram stuff ===================================================================================================

async def message_user(chat_id: int, msg: str, context: telegram.ext.ContextTypes.DEFAULT_TYPE, parse_mode=ParseMode.MARKDOWN):
    logging.debug(f"sending message to chat {chat_id}: {msg}")
    if parse_mode is None:
        await context.bot.send_message(chat_id=chat_id,
                                       text=msg,
        )  # reply_to_message_id=update.message.message_id
    else:
        await context.bot.send_message(chat_id=chat_id,
                                       text=msg,
                                       parse_mode=parse_mode,
        )  # reply_to_message_id=update.message.message_id

async def start(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE):
    logging.debug("adding new chat")
    await message_user(update.effective_chat.id, "welcome!", context)
    

def reset_conversation(chat_id):
    conversations[chat_id] = [SYSTEM_PROMPT]  # reset conversation

async def new_conversation(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE):
    reset_conversation(update.effective_chat.id)
    logging.debug("new conversation started")
    await message_user(update.effective_chat.id, "new conversation started", context)

async def record_user_message(update: telegram.Update, process_msg=lambda msg: msg.text):
    current_chat_id = update.effective_chat.id

    chat_already_exists = current_chat_id in conversations
    if not chat_already_exists:
        reset_conversation(current_chat_id)

    conversations[current_chat_id].append({"role": "user", "content": process_msg(update.message)})
    logging.info(f"conversations: {conversations}")
    return chat_already_exists

async def warn_chat_missing(context: telegram.ext.ContextTypes.DEFAULT_TYPE, current_chat_id: int, chat_already_exists: bool):
    if not chat_already_exists:
        logging.info("chat missing!")
        await message_user(current_chat_id, "looks like i've lost the previous conversation! i won't remember anything earlier than your most recent message.", context)

async def respond_conversation(current_chat_id: int, context: telegram.ext.ContextTypes.DEFAULT_TYPE, model=MODEL):
    try:
        response = query_model(conversations[current_chat_id], model=model)
    
    except openai.error.InvalidRequestError as e:
        logging.error(f"invalid request error: {e}")
        if "Please reduce the length of the messages" in e.user_message:
            logging.info(f"model failed for length; trying long context model")
            if model == MODEL:
                response = query_model(conversations[current_chat_id], model=LONG_MODEL)
            elif model == FAST_MODEL:
                response = query_model(conversations[current_chat_id], model=LONG_FAST_MODEL)
            else:
                logging.error("invalid model")
                raise e

    except Exception as e:
        # TODO: should maybe give some option to remove the last response so it doesn't ruin the whole conversaion,
        #   if that turns out to be a problem
        logging.error(e)
        await message_user(current_chat_id, f"model failed with error: {e}", context, parse_mode=None)
        return
    conversations[current_chat_id].append({"role": "assistant", "content": response})

    await message_user(current_chat_id, response, context)

async def regular_message(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE, process_msg=lambda msg: msg.text, model=MODEL):
    logging.info(f"update received at {time.time()}: {update}")
    
    chat_already_exists = await record_user_message(update, process_msg)
    current_chat_id = update.effective_chat.id
    await warn_chat_missing(context, current_chat_id, chat_already_exists)
    await respond_conversation(current_chat_id, context, model)

def is_turbo(text: str):
    return text.startswith(f"/{TURBO_COMMAND}")

def remove_turbo(text: str):
    return text[len(TURBO_COMMAND) + 2:]

async def turbo_message(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE):
    logging.info("turbo message received")
    await regular_message(update, context, process_msg=lambda msg: remove_turbo(msg.text), model=FAST_MODEL)

async def no_response(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE):
    logging.info(f"message with no response requested received at {time.time()}: {update}")
    def removenoresponsecommand(msg):
        return msg.text[len(NO_RESPONSE_COMMAND) + 2:]

    chat_already_exists = await record_user_message(update, process_msg=removenoresponsecommand)
    await warn_chat_missing(context, update.effective_chat.id, chat_already_exists)

async def record_document(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE):
    content = await read_document(update.message, context)

    chat_already_exists = await record_user_message(update, process_msg=lambda msg: format_document(msg.document.file_name, content, msg.caption))
    await warn_chat_missing(context, update.effective_chat.id, chat_already_exists)

async def handle_document(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE):
    logging.info(f"document received at {time.time()}: {update}")
    await record_document(update, context)
    
    current_chat_id = update.effective_chat.id
    await warn_chat_missing(context, current_chat_id, True)

    # if the document has a caption, respond to the document; otherwise, just acknowledge receipt
    if update.message.caption is not None:
        logging.info("document with caption received")
        if is_turbo(update.message.caption):
            logging.info("turbo document")
            model = FAST_MODEL
        else:
            model = MODEL
        await respond_conversation(current_chat_id, context, model=model)
    else:
        logging.info("document without caption received")
        await message_user(current_chat_id, "document received", context)


async def read_document(msg: telegram.Message, context: telegram.ext.ContextTypes.DEFAULT_TYPE):
    # check that the effective attachment is a file rather than a photo, poll, etc
    
    logging.debug(f"msg.effective_attachment: {msg.effective_attachment}")
    if msg.document is None:
        logging.info("message is not a document")
        await message_user(msg.chat_id, "sorry, i can only read documents", context)
        raise Exception("message is not a document")
    
    new_file = await msg.effective_attachment.get_file()

    # save the file locally to the tmp directory with a random filename
    random_string = ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    local_path = os.path.join("/tmp", random_string)
    
    await new_file.download_to_drive(custom_path=local_path)

    # read in the file as text
    with open(local_path, 'r') as document:
        content = document.read()

    # delete the file
    os.remove(local_path)

    # return the content
    return content


def format_document(file_name: str, content: str, caption: typing.Optional[str]):
    content = f"Document title: {file_name}\nDocument content: {content}"
    if caption is not None:
            if is_turbo(caption):
                logging.info("turbo document")
                content += f"\nDocument caption: {remove_turbo(caption)}"
            else:
                content += f"\nDocument caption: {caption}"
    return content

async def unsupported_message(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE):
    await message_user(update.effective_chat.id, "sorry, i don't know how to handle that type of message", context)

async def unsupported_message_handler(update: telegram.Update, context: telegram.ext.ContextTypes.DEFAULT_TYPE):
    logging.info(f"unsupported message received at {time.time()}: {update}")
    await unsupported_message(update, context)

# === OpenAI stuff =====================================================================================================

def query_model(previous_messages, model=MODEL):
    response = openai.ChatCompletion.create(model=model, messages=previous_messages)
    logging.info(f"got response: {response}")
    return response['choices'][0]['message']['content']


if __name__ == '__main__':
    dotenv.load_dotenv()

    openai.api_key = os.getenv("OPENAI_API_KEY")
    BOT_TOKEN = os.getenv("BOT_TOKEN")

    application = telegram.ext.ApplicationBuilder().token(BOT_TOKEN).build()

    start_handler = telegram.ext.CommandHandler('start', start)
    new_conversation_handler = telegram.ext.CommandHandler('new_conversation', new_conversation)
    turbo_handler = telegram.ext.CommandHandler(TURBO_COMMAND, turbo_message)
    no_response_handler = telegram.ext.CommandHandler(NO_RESPONSE_COMMAND, no_response)
    regular_message_handler = telegram.ext.MessageHandler(telegram.ext.filters.TEXT & (~telegram.ext.filters.COMMAND), regular_message)
    document_handler = telegram.ext.MessageHandler(telegram.ext.filters.ATTACHMENT, handle_document)
    error_handler = telegram.ext.MessageHandler(telegram.ext.filters.ALL, unsupported_message)

    application.add_handler(start_handler)
    application.add_handler(regular_message_handler)
    application.add_handler(new_conversation_handler)
    application.add_handler(turbo_handler)
    application.add_handler(no_response_handler)
    application.add_handler(document_handler)
    application.add_handler(error_handler)

    application.run_polling()
