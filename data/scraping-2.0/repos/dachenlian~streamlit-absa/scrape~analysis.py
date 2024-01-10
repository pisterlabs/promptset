from collections import Counter
from pathlib import Path
from string import Template
from typing import Callable

from loguru import logger
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import openai
import tiktoken
from wordcloud import WordCloud

from scrape.absa.aspects import FINANCIAL_ASPECTS, MOVIE_ASPECTS
from scrape.absa.prompts import GET_ABSA_FINANCE_PROMPT, GET_ABSA_MOVIE_PROMPT
from scrape.types import OpenAIModel

enc = tiktoken.get_encoding("cl100k_base")
BASE = Path(__file__).resolve().parent.parent
font_path = str(BASE / "data/fonts/SourceHanSerifK-Light.otf")


def get_summary(
    client: openai.Client,
    texts: list[str],
    base_prompt: Template,
    main_body: str | None = None,
    max_length: int = 2000,
    model_name: OpenAIModel = OpenAIModel.GPT4,
) -> str:
    from scrape.utils import call_model

    _texts = [f"Text {i}:\n{t}" for i, t in enumerate(texts, start=1)]
    texts = []
    length = 0
    for t in _texts:
        length += len(enc.encode(t))
        texts.append(t)
        if length > max_length:
            break

    prompt = base_prompt.substitute(text=texts)
    if main_body:
        prompt = "main_body: " + main_body + "\n\n" + prompt
    messages = [{"role": "user", "content": prompt}]

    response = call_model(client, messages, model_name)

    return response


def create_wordcloud(
    freq: Counter, width: int = 1280, height: int = 720, font_path: str | None = font_path
) -> None:
    fig = plt.figure(figsize=(10, 10))
    wc = WordCloud(
        font_path=font_path,
        background_color="white",
        max_words=1000,
        width=width,
        height=height,
    )
    wc.generate_from_frequencies(freq)
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")


def create_word_count(
    texts: list[str],
    stop: list[str] | None = None,
) -> Counter:
    c = Counter()
    for t in texts:
        c.update(t.lower().split())

    if stop:
        for w in stop:
            c.pop(w, None)

    return c
