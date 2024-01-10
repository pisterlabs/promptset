import nltk
import os
import openai
from random import choice

from aiogram import types

from .. import handlers
from ..config import dp, GTRANSLATE_LANGS
from ..config.languages import LANGS, Language
from ..config.i18n import i18n

nltk.download('punkt')  # Download the punkt tokenizer

# Ensure you've set the OPENAI_API_KEY as an environment variable
gpt_model = api_key="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"

async def cleared_translate(text: str, tgt_lang: str) -> str:
    text_parts = nltk.sent_tokenize(text)  # Split the text into sentences
    translations = []  # List to hold the translated parts

    for part in text_parts:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Ты сумасшедший переводчик!"},
                {"role": "user", "content": f"Переведи на {tgt_lang} следующий текст: {part}"},
            ],
            max_tokens=2048,
            temperature=1,
        )
        translations.append(response['choices'][0]['message']['content'].strip())

    return ' '.join(translations)

def get_lang_emoji_by_name(lang_name: str) -> str:
    return LANGS.get(lang_name, Language(lang_name, lang_name)).emoji

@dp.message_handler(commands=["crazy", "crazy2", "c"])
@handlers.get_text
async def crazy_translator(message: types.Message, text: str):
    msg = await message.reply("⏳")

    user_lang = (
        user_lang
        if (user_lang := await i18n.get_user_locale()) in ("uk", "ru", "en")
        else "ru"
    )

    langs = []

    for __ in range(7):
        lang = choice(
            tuple(
                filter(
                    lambda x: x not in langs,
                    GTRANSLATE_LANGS,
                )
            )
        )
        lang = "uk" if lang == "ua" else lang
        langs.append(lang)

        text = await cleared_translate(text, tgt_lang=lang)

    langs.append(user_lang)

    final_translation = await cleared_translate(text, tgt_lang=user_lang)

    await msg.edit_text(
        final_translation or "None",
        disable_web_page_preview=True,
    )

    if message.get_command(pure=True).endswith("2"):
        await message.answer("".join(map(get_lang_emoji_by_name, langs)))
