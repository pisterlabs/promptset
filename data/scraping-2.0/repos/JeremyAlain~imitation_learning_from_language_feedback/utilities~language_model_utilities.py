import os
import re
import string
from typing import Any, Generator, List, Optional

from dotenv import load_dotenv
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
import openai
import unidecode


def set_openai_key_to_environment_variable() -> None:
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    assert (
        openai_api_key is not None
    ), "For GPT3 access you need to set the OPENAI_API_KEY in your environment variables. See the README for help"
    openai.api_key = openai_api_key


def list_to_batches_generator(
    list: List[Any], batch_size: int
) -> Generator[List[Any], None, None]:
    list_length = len(list)
    for i in range(0, list_length, batch_size):
        yield list[i : min(i + batch_size, list_length)]


def list_of_completions_to_completions_per_prompt(
    list_of_generations: List[str],
    number_of_prompts: int,
    number_of_generations_per_prompt: int,
) -> List[List[str]]:
    assert (
        len(list_of_generations) == number_of_prompts * number_of_generations_per_prompt
    )
    generations_per_prompt = []
    for i in range(number_of_prompts):
        generations_per_prompt.append(
            list_of_generations[
                i
                * number_of_generations_per_prompt : (i + 1)
                * number_of_generations_per_prompt
            ]
        )
    return generations_per_prompt


class TextNormalizer:
    """
        Text Normalization for NLP
        -removes extra whitespace within text
        -converts unicode to ascii
        -converts to lowercase
        -remove leading or trailing whitespace
        -expands contractions
        -tokenizes sentences and words
        -removes punctuation
        -lemmatizes words
        -removes stopwords
        Taken from https://gist.github.com/lvngd/3695aac64461de2cfb9d50bb11d5fbb3 and slightly adapted.
    """

    def __init__(self) -> None:
        nltk.download("stopwords")
        nltk.download("punkt")
        nltk.download("wordnet")
        self.lemmatizer = WordNetLemmatizer()
        self.punctuation_table = str.maketrans("", "", string.punctuation)
        self.stop_words = set(stopwords.words("english"))

    def normalize_text(self, text: str) -> List[str]:
        normalized_sentences = []
        text = re.sub(" +", " ", text)
        text = unidecode.unidecode(text)
        text = text.lower()
        # text = contractions.fix(text)
        sentences = sent_tokenize(text)
        for sentence in sentences:
            # remove punctuation
            sentence = sentence.translate(self.punctuation_table)
            # strip leading/trailing whitespace
            sentence = sentence.strip()
            words = word_tokenize(sentence)
            # lemmatize and remove stopwords
            # filtered = [self.lemmatizer.lemmatize(word) for word in words if word not in self.stop_words]
            new_sentence = " ".join(words)
            normalized_sentences.append(new_sentence)
        return normalized_sentences


def texts_have_forbidden_substrings(
    texts: List[str], forbidden_substrings: Optional[List[str]]
) -> bool:
    if forbidden_substrings is None:
        for text in texts:
            if len(text.strip()) == 0:
                return True
    else:
        for text in texts:
            for forbidden_substring in forbidden_substrings:
                if forbidden_substring in text or len(text.strip()) == 0:
                    return True
    return False
