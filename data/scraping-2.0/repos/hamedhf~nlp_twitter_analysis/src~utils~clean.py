import re
import string
import time

import nltk
import openai
from cleantext.clean import remove_emoji as clean_text_remove_emoji
from dotenv import load_dotenv
from hazm import Normalizer, stopwords_list
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from openai.error import ServiceUnavailableError, APIError

from .constants import TOPICS, get_api_base_url, get_api_key


def replace_emojis(text):
    # Happy
    grin = 'خنده'
    laugh = 'خنده'
    happy = 'خوشحال'
    _text = re.sub(":D", grin, text)
    _text = re.sub(" (x|X)D", laugh, _text)
    _text = re.sub(":\)+", happy, _text)

    # Sad
    sad = 'ناراحت'
    annoyed = 'رنجیده'
    _text = re.sub(":\(+", sad, _text)
    _text = re.sub("-_+-", annoyed, _text)
    return _text


def remove_emojis(text):
    _text = clean_text_remove_emoji(text)
    return _text


def remove_url(text):
    _text = re.sub(r"https?:\S+", '', text)
    return _text


def remove_punc(text):
    _text = text.translate(str.maketrans('', '', string.punctuation))
    persian_virgol = '،'  # noqa
    _text = _text.replace(persian_virgol, ' ')
    return _text


def remove_stopwords(text):
    # TODO: Check if this is a good idea or not.
    # link: https://medium.com/analytics-vidhya/a-simple-yet-effective-way-of-text-cleaning-using-nltk-4f90a8ff21d4
    # text_data = [wl.lemmatize(word) for word in text_data if not word in set(stopwords.words('english'))]
    pass


def remove_numbers(text):
    _text = re.sub(r'\d+', '', text)
    return _text


def remove_hashtags(text):
    _text = re.sub(r'#\S+', '', text)
    return _text


def remove_mentions(text):
    _text = re.sub(r'@\S+', '', text)
    return _text


def remove_duplicate_spaces(text):
    _text = " ".join(text.split())
    return _text


def should_be_translated(text: str, percentage_thresh: float = 0.4) -> bool:
    nltk.download('stopwords')
    nltk.download('punkt')
    nltk.download('wordnet')

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords from the token list
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word.casefold() not in stop_words]

    # Find the English words
    english_words = []
    for word in filtered_tokens:
        if nltk.corpus.wordnet.synsets(word):
            english_words.append(word)

    english_words_len = len(english_words)
    total_words_len = len(filtered_tokens)
    percentage = english_words_len / total_words_len
    print(f"percentage of English words: {percentage}")

    if percentage >= percentage_thresh:
        print("Need to translate")
        return True
    else:
        print("No need to translate")
        return False


def translate_english_to_persian(
        api_key: str,
        api_base_url: str,
        text: str,
        percentage_thresh: float = 0.4,
        sleep_seconds: int = 30
) -> str:
    openai.api_key = api_key
    openai.api_base = api_base_url

    if not should_be_translated(text, percentage_thresh):
        return text

    messages = [
        {
            "role": "system",
            "content": f"Translate the following text from English to Persian without changing Persian words"
        },
        {"role": "user", "content": f"{text}"},

    ]

    while True:
        try:
            print("*" * 100)
            print(messages)
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.1
            )
            print(response)
            print("*" * 100)

            text = str(response['choices'][0]['message']['content']).strip()
            print(f"Translated text: {text}")
            break
        except KeyError:
            print(f"KeyError occurred. Skipping translation for this text: {text}")
        except (ServiceUnavailableError, APIError):
            print(f"ServiceUnavailableError or APIError occurred. Sleeping for {sleep_seconds} seconds")
            time.sleep(sleep_seconds)

    print(f"Sleeping for {sleep_seconds} seconds to avoid rate limit")
    time.sleep(sleep_seconds)
    return text


def clean_text(text) -> tuple[str, str]:
    _punc_text = remove_numbers(
        remove_mentions(
            remove_hashtags(
                remove_duplicate_spaces(
                    remove_url(
                        remove_emojis(text)
                    )
                )
            )
        )
    )

    _text = remove_punc(_punc_text)
    normalizer = Normalizer()
    _text = normalizer.normalize(_text)

    return _text, _punc_text


def get_clean_label(input_label: str) -> str:
    for value in TOPICS.values():
        if value in input_label:
            label = value
            break
    else:
        label = 'other'
    return label


if __name__ == "__main__":
    uncleaned_persian_text = "امروز با بچه‌ها میخوایم بریم بیرون و بععععدش بریم سینما :D https://t.co/1234567890"
    cleaned_text = clean_text(uncleaned_persian_text)
    print(cleaned_text)

    print(stopwords_list())

    nltk.download('stopwords')
    print(stopwords.words('english'))

    print(remove_url('https://github.com/roshan-research/hazm hi there'))
    print(remove_url('https hi there'))

    load_dotenv('../.env')
    api_key = get_api_key()
    api_base_url = get_api_base_url()

    tweet1 = 'نینا آهنگی داره stars Stars they come and go  they come fast they come slow که در اپیزود everyone has a story بوجک هم استفاده شده و حقا که ترکیب اوون صحنه‌ها‌و آهنگ منقلب کنندست'  # noqa
    tweet2 = 'These boots are made for walkin'  # noqa
    print(translate_english_to_persian(api_key, api_base_url, tweet1))
    print(translate_english_to_persian(api_key, api_base_url, tweet2))

    # download model from: https://github.com/roshan-research/hazm
    # tagger = POSTagger(model='resources/pos_tagger.model')
    # tagger.tag(word_tokenize('ما بسیار کتاب می‌خوانیم'))
