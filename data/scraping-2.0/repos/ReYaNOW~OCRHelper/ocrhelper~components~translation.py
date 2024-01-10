import openai
from deep_translator import GoogleTranslator
from loguru import logger
import keyring


def translation(text, from_lang, to_lang, translator='Google Translator'):
    logger.info(f'Перевод при помощи {translator}')
    match translator:
        case 'Google':
            if len(from_lang) == 1 or from_lang not in ('ru', 'en'):
                from_lang_checked = 'auto'
            else:
                from_lang_checked = from_lang[0]

            translator_obj = GoogleTranslator(
                source=from_lang_checked,
                target=to_lang,
            )
            return translator_obj.translate(text=text)

        case 'GPT':
            from_lang_conv = lang_convert(from_lang)
            to_lang_conv = lang_convert(to_lang)
            return gpt_request(text, from_lang_conv, to_lang_conv)

        case 'GPT Stream':
            from_lang_conv = lang_convert(from_lang)
            to_lang_conv = lang_convert(to_lang)
            return gpt_request(
                text, from_lang_conv, to_lang_conv, use_stream=True
            )


def gpt_request(text, from_lang, to_lang, use_stream=False):
    if from_lang == to_lang and not use_stream:
        return text

    request = f'Please translate the user message from {from_lang} to\
     {to_lang}. Make the translation sound as natural as possible.\
      In answer write ONLY translation.\n\n {text}'

    response = openai.ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': request}],
        stream=use_stream,
        api_key=keyring.get_password("system", "GPT_API_KEY"),
    )
    if use_stream:
        return response
    return response['choices'][0]['message']['content']


def lang_convert(language):
    match language:
        case ['en']:
            return 'english'
        case ['ru']:
            return 'russian'
        case ['en', 'ru']:
            return 'english and russian'
        case ['ru', 'en']:
            return 'english and russian'
        case _:
            return language
