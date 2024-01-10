from dotenv.main import load_dotenv
import os
import logging
import openai
import replicate
from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove
from telegram.ext import (ApplicationBuilder, 
                          ContextTypes, 
                          CommandHandler, 
                          filters, 
                          MessageHandler,
                          ConversationHandler)

import urllib.request
from moviepy.editor import AudioFileClip

load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

### LOGGING
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

### START MENU

MODE, CHAT, MENU_LANGUAGE, TEXT_TO_TRANSLATE, MENU_PHOTO_CAPTION, PHOTO_CAPTION, START_VOCAB_PIC, VOCAB_PIC, ANSWER_VOCAB_PIC = range (9)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    reply_keyboard = [["Translation", "Photo Caption", 'Vocabulary Pics'],
                      ["Generic Chat"]]
    await update.message.reply_text(
        "Hello, I am your Finnish languague trainer.\n\nYou can send a text to translate, a photo, a voice message or just simply chat\n\nChoose the chat mode:",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True, input_field_placeholder="Chat or Translation?"
        ),
    )
    return MODE

### GENERIC CHAT


messages = [{"role": "system", "content": "You are a Finnish language teacher named Anna"}]

async def generic_chat(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:

    logger.info("User: %s", update.message.text)

    messages.append({"role": "user", "content": update.message.text})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]

    logger.info("Response from ChatGPT: %s", ChatGPT_reply)

    await update.message.reply_text(text=f"*[Bot]:* {ChatGPT_reply}", parse_mode= 'MARKDOWN')
    messages.append({"role": "assistant", "content": ChatGPT_reply})

    return CHAT

### TRANSLATION

async def translate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    '''Starts the conversation and asks the user about which language they want to translate'''
    reply_keyboard = [['🇬🇧 English', '🇫🇮 Finnish','🇮🇹 Italian']]

    logger.info('User chose Translation')

    await update.message.reply_text(
        "Which destination language do you want to translate to?",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True, input_field_placeholder="Choose the destination language"
        ),
    )

    return MENU_LANGUAGE


async def _start_translation(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Stores the selected language and ask for the text to translate"""
    translation_language = update.message.text

    logger.info("Language chosen: %s", translation_language)

    user_data = context.user_data
    user_data['translation_language'] = translation_language

    await update.message.reply_text(
        "Type the text you want to translate to " + translation_language,
        reply_markup=ReplyKeyboardRemove(),
    )

    return TEXT_TO_TRANSLATE

async def text_to_translate(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    '''Get the text and return the translation'''
    logger.info("Text to translate: %s", update.message.text)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{'role':'system','content': 'Translate the text to ' + context.user_data['translation_language']},
                  {'role':'user','content': update.message.text}
                  ]
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    logger.info("Response from ChatGPT: %s", ChatGPT_reply)
    await update.message.reply_text(text=f"*[Translation]:* {ChatGPT_reply}", parse_mode= 'MARKDOWN')

    return TEXT_TO_TRANSLATE

### VOICE MESSAGE
async def voice_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    logger.info("User sent a voice message")
    await update.message.reply_text("I've received a voice message! Please give me a second to respond :)")
    file_info = await context.bot.getFile(update.message.voice.file_id)
    urllib.request.urlretrieve(file_info.file_path, "voice_message.oga")
    audio_clip = AudioFileClip("voice_message.oga")
    audio_clip.write_audiofile("voice_message.mp3")
    audio_file = open("voice_message.mp3", "rb")
    transcript = openai.Audio.transcribe("whisper-1", audio_file).text
    
    await update.message.reply_text(text=f"*[You]:* _{transcript}_", parse_mode='MARKDOWN')
    messages.append({"role": "user", "content": transcript})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]
    await update.message.reply_text(text=f"*[Bot]:* {ChatGPT_reply}", parse_mode='MARKDOWN')
    messages.append({"role": "assistant", "content": ChatGPT_reply})

### PHOTO CAPTION
async def _start_photo_caption(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    logger.info("User started photo caption")
    await update.message.reply_text(
        "You can send me photo and I will give it a caption",
        reply_markup=ReplyKeyboardRemove(),
    )
    return PHOTO_CAPTION

async def photo_caption(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    logger.info("User sent a photo")
    await update.message.reply_text("I've received a photo, let me give you the caption about it")
    file_info = await context.bot.getFile(update.message.photo[3].file_id) #0 for thumbnail and 3 for bigger size
    urllib.request.urlretrieve(file_info.file_path, 'photo.jpg')

    caption_ENG = replicate.run(
    "salesforce/blip:2e1dddc8621f72155f24cf2e0adbde548458d3cab9f00c0139eea840d0ac4746",
    input={"image": open("photo.jpg", "rb")}
    )
    translation = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{'role':'system','content': 'Translate the text to Finnish and Italian, put it in this format: ENG: <original text> \n FI: <text> \n IT: <text>'},
                  {'role':'user','content': caption_ENG}
                  ]
    )
    caption = translation["choices"][0]["message"]["content"]
    logger.info("Response from ChatGPT: %s", caption)
    await update.message.reply_text(text=f"{caption}", parse_mode= 'MARKDOWN')

    return PHOTO_CAPTION

### VOCABULARY PICS
async def _start_vocab_pic(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    reply_keyboard = [['OK send me the pic']]
    await update.message.reply_text(
        "In this exercise, you will receive a pic and try to describe it in Finnish",
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True, input_field_placeholder="I am ready!"
        ),
    )
    return VOCAB_PIC

async def vocab_pic(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text("Mikä tämä on?")
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": 'Randomly choose a word from a B2 language level vocabulary list, which describes things around a person in daily live'}]
    )

    new_word = response["choices"][0]["message"]["content"]
    logger.info("The new word is " + new_word)

    # Generate image
    response = openai.Image.create(
        prompt= f'Realistic photo of {new_word}, simple background',
        n=1,
        size="512x512")
    image_url = response['data'][0]['url']
    
    await context.bot.send_photo(chat_id=context._chat_id, photo=image_url)

    logger.info("Bot have sent the picture to user")


    # Translate the new word to Finnish
    response_FI = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{'role':'system','content': 'Translate this word to Finnish: '},
                  {'role':'user','content': new_word}
                  ]
    )
    vocab_word_FI = response_FI["choices"][0]["message"]["content"]

    logger.info("The word in Finnish is: " + vocab_word_FI)

    user_data = context.user_data
    user_data['vocab_word'] = new_word
    user_data['vocab_word_FI'] = vocab_word_FI

    return ANSWER_VOCAB_PIC

async def _answer_vocab_pic(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    vocab_word = context.user_data['vocab_word']
    vocab_word_FI = context.user_data['vocab_word_FI']

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{'role':'system','content': 'Check if the word is matched or a synonym with the correct word which is' + vocab_word_FI +'. If it is correct, say congratulation, if not correct, show the correct word and explain it in English'},
                  {'role':'user','content': update.message.text}
                  ]
    )
    ChatGPT_reply = response["choices"][0]["message"]["content"]

    await update.message.reply_text(text=f"{ChatGPT_reply}\nAnswer: {vocab_word_FI} ( {vocab_word} )", parse_mode= 'MARKDOWN')

    reply_keyboard = [['OK next one!']]
    await update.message.reply_text('Do you want to continue?',
        reply_markup=ReplyKeyboardMarkup(
            reply_keyboard, one_time_keyboard=True, input_field_placeholder="Let's continue"
        ),
    )
    return START_VOCAB_PIC

### End the chat
async def quit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Quit chat"""
    logger.info("User quitted the chat.")
    await update.message.reply_text(
        "You quitted chat", reply_markup=ReplyKeyboardRemove()
    )

    return ConversationHandler.END

def main() -> None:
    application = ApplicationBuilder().token(os.environ['TELEGRAM_BOT_TOKEN']).build()
    
    start_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            MODE: [
                MessageHandler(filters.Regex('^(Generic Chat)$'), generic_chat),
                MessageHandler(filters.Regex('^(Translation)$'), translate),
                MessageHandler(filters.Regex('^(Photo Caption)$'), _start_photo_caption),
                MessageHandler(filters.Regex('^(Vocabulary Pics)$'), _start_vocab_pic)
            ],
            CHAT: [MessageHandler(filters.TEXT & ~filters.COMMAND, generic_chat),
                   MessageHandler(filters.VOICE & ~filters.COMMAND, voice_message),
                   MessageHandler(filters.PHOTO & ~filters.COMMAND, photo_caption)
                   ],
            MENU_LANGUAGE: [MessageHandler(filters.Regex("^(🇫🇮 Finnish|🇬🇧 English|🇮🇹 Italian)$"), _start_translation)],
            TEXT_TO_TRANSLATE: [MessageHandler(filters.TEXT & ~filters.COMMAND, text_to_translate)],
            PHOTO_CAPTION: [MessageHandler(filters.PHOTO & ~filters.COMMAND, photo_caption)],
            START_VOCAB_PIC: [MessageHandler(filters.Regex('^(OK next one!)$'), _start_vocab_pic)],
            VOCAB_PIC: [MessageHandler(filters.TEXT & ~filters.COMMAND, vocab_pic)],
            ANSWER_VOCAB_PIC: [MessageHandler(filters.TEXT & ~filters.COMMAND, _answer_vocab_pic)],
        },
        fallbacks=[CommandHandler("quit", quit),
                   CommandHandler('menu', start),
                   CommandHandler('caption', _start_photo_caption)],
    )

    application.add_handler(start_handler)

    application.run_polling()

if __name__ == '__main__':
    main()