#!/usr/bin/python3

"""
            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
                    Version 2, December 2004
 Copyright (C) 2023 Christian "veeso" Visintin
 Everyone is permitted to copy and distribute verbatim or modified
 copies of this license document, and changing it is allowed as long
 as the name is changed.
            DO WHAT THE FUCK YOU WANT TO PUBLIC LICENSE
   TERMS AND CONDITIONS FOR COPYING, DISTRIBUTION AND MODIFICATION
  0. You just DO WHAT THE FUCK YOU WANT TO.
"""

from logging import info, debug, error
from os import environ, unlink
from sys import argv
import logging
import openai
from tempfile import NamedTemporaryFile

from .speech.tts import TTSError
from .speech.google_translate import GoogleTranslateTTS

try:
    from telegram.ext._updater import Updater
    from telegram._update import Update
    from telegram.ext._callbackcontext import CallbackContext
    from telegram.ext._commandhandler import CommandHandler
except ImportError:
    from telegram.ext.updater import Updater
    from telegram.update import Update
    from telegram.ext.callbackcontext import CallbackContext
    from telegram.ext.commandhandler import CommandHandler

CHATGPT_API_KEY = environ.get("CHATGPT_API_KEY", "")
LOG_LEVEL = environ.get("LOG_LEVEL", "info")
TELEGRAM_API_KEY = environ.get("TELEGRAM_API_KEY", "")


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=get_loglevel(LOG_LEVEL),
    )

    updater = Updater(TELEGRAM_API_KEY, use_context=True)
    info(f"initialized telegram updater {TELEGRAM_API_KEY}")
    updater.dispatcher.add_handler(CommandHandler("start", start))
    updater.dispatcher.add_handler(CommandHandler("padrepio", padrepio))
    updater.dispatcher.add_handler(CommandHandler("padrepio-dice", padrepio_dice))
    updater.dispatcher.add_handler(CommandHandler("help", help))
    info("starting telegram bot")
    # Start the Bot
    info("starting bot")
    updater.start_polling()
    updater.idle()
    exit(0)


def send_request(text: str) -> str:
    openai.api_key = CHATGPT_API_KEY

    messages = [
        {
            "role": "system",
            "content": "Sei Padre Pio, noto anche come Pio da Pietrelcina. Il tuo vero nome è Francesco Forgione, nato il 25 maggio 1887 e morto il 23 settembre 1968. Sei stato un presbitero e mistico italiano. Nel 2002 sei stato proclamato santo da Giovanni Paolo II. Ti piace ringraziare gli utenti per le domande e pregare, mentre ti rifiuti di rispondere a chi ti bestemmia contro.",
        },
        {
            "role": "user",
            "content": f'Rispondi a "{text}" considerando l\'informazione data, in italiano come se fossi Padre Pio parlando in prima persona.',
        },
    ]

    chat = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=messages)
    reply = chat.choices[0].message.content

    return reply


def start(update: Update, _: CallbackContext):
    update.message.reply_text(
        "Buongiorno, benvenuto. Io sto pregando e prestando servizio ai fratelli e sorelle bisognosi. La pace di Gesù sia con te. Chiedimi qualsiasi cosa con /padrepio <domanda>"
    )


def padrepio(update: Update, context: CallbackContext):
    text: str = (
        update.message.text.replace("/padrepio", "")
        .replace(context.bot.name, "")
        .strip()
    )
    debug(f"text: {text}")
    if len(text) == 0:
        context.bot.send_message(
            chat_id=update.message.chat_id, text="Ti ascolto figliolo"
        )
        return
    try:
        debug("sending request to padre pio")
        answer = send_request(text)
    except Exception as err:
        error(f"failed to get openai response: {err}")
        return reply_err(update, f"Non riesco a contattare Padre Pio: {err}")
    debug("got an answer from padre pio")
    context.bot.send_message(chat_id=update.message.chat_id, text=answer)


def padrepio_dice(update: Update, context: CallbackContext):
    text: str = (
        update.message.text.replace("/padrepio-dice", "")
        .replace(context.bot.name, "")
        .strip()
    )
    debug(f"text: {text}")
    if len(text) == 0:
        context.bot.send_message(
            chat_id=update.message.chat_id, text="Ti ascolto figliolo"
        )
        return
    try:
        debug("sending request to padre pio")
        answer = send_request(text)
    except Exception as err:
        error(f"failed to get openai response: {err}")
        return reply_err(update, f"Non riesco a contattare Padre Pio: {err}")
    debug("got an answer from padre pio")
    # get tts for answer
    tts_engine = GoogleTranslateTTS()
    try:
        debug(f"getting speech for {answer}")
        audio = tts_engine.get_speech(answer)
    except TTSError as err:
        error(f"failed to get tts speech: {err}")
        return reply_err(update, "Padre Pio non può parlare in questo momento")

    debug("correctly got the audio from tts engine")
    # writing audio to tempfile
    with NamedTemporaryFile("w+b", suffix=".ogg", delete=False) as f:
        audio.export(f.name, "ogg")
        f.close()
        debug(f"audio exported to {f.name}")
        # sending document
        debug("sending voice message...")
        context.bot.send_voice(
            chat_id=update.message.chat_id,
            voice=open(f.name, "rb"),
            duration=audio.duration_seconds,
        )
        info("audio file sent")
        unlink(f.name)
        debug("file removed")


def reply_err(update: Update, text: str):
    update.message.reply_text(text)


def help(update: Update, _: CallbackContext):
    update.message.reply_text(
        """/padrepio <testo> - chiedi consiglio a padre pio
/padrepio-dice <testo> - chiedi consiglio a Padre Pio, ma ti risponde parlando
/help - mostra questo messaggio"""
    )


def get_loglevel(level: str) -> int:
    try:
        return {
            "info": logging.INFO,
            "error": logging.ERROR,
            "debug": logging.DEBUG,
            "warn": logging.WARN,
        }.get(level)
    except Exception:
        return logging.ERROR


if __name__ == "__main__":
    args = argv[1:]
    if len(args) == 0:
        main()
    else:
        prompt = " ".join(args)
        print(send_request(prompt))
