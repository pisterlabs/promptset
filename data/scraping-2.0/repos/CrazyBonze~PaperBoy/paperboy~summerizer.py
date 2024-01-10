import asyncio
from textwrap import shorten

import openai
import pysrt
from settings import settings
from sumy.nlp.stemmers import Stemmer
from sumy.nlp.tokenizers import Tokenizer
from sumy.parsers.plaintext import PlaintextParser
from sumy.summarizers.lsa import LsaSummarizer as Summarizer
from sumy.utils import get_stop_words

LANGUAGE = "english"
SENTENCES_COUNT = 5


async def new_summarize(text):
    openai.api_key = settings.openai_api_key
    messages = [
        {"role": "system", "content": settings.system_message},
        {"role": "user", "content": text},
    ]
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: openai.ChatCompletion.create(
            model=settings.openai_model, messages=messages
        ),
    )
    ai_message = response["choices"][0]["message"]["content"]
    return ai_message


def _summarize(text, sentences=SENTENCES_COUNT, language=LANGUAGE):
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    stemmer = Stemmer(LANGUAGE)

    summarizer = Summarizer(stemmer)
    summarizer.stop_words = get_stop_words(LANGUAGE)

    return shorten(
        " ".join([str(s) for s in summarizer(parser.document, SENTENCES_COUNT)]),
        width=1800,
        placeholder="...",
    )


async def summarize(text, sentences=SENTENCES_COUNT, language=LANGUAGE):
    loop = asyncio.get_running_loop()
    done = loop.run_in_executor(None, _summarize, text, sentences, language)
    return await done
