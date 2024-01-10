from typing import List, Optional
import os
import requests
from io import StringIO

from dotenv import load_dotenv
import youtube_dl
import webvtt
import openai
from googletrans import Translator

import config as cfg

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


def get_subtitles_link(video_url: str, language: str) -> Optional[str]:
    ydl_opts = {
        "dump-json": True,
        "writesubtitles": True,
        "writeautomaticsub": True,
        "youtube_include_dash_manifest": False,
    }
    try:
        with youtube_dl.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(video_url, download=False)
            if not info_dict["formats"]:
                print("Status : Something went wrong retry or video is unavailable")
                return None
    except Exception:
        print("Error : Check your Internet Connection or Url.")
        return None

    subtitles = info_dict.get("subtitles")

    if subtitles:
        subtitle = subtitles.get(language)
        if subtitle:
            for fmt in subtitle:
                if fmt["ext"] == "vtt":
                    sub_dlink = fmt["url"]
                    return sub_dlink
    return None


def get_subtitles(video_url: str, language: str) -> Optional[str]:
    subtitle_url = get_subtitles_link(video_url=video_url, language=language)

    if subtitle_url:
        response = requests.get(subtitle_url)

        if response.status_code == 200:
            subtitles = response.text
            print("Subtitles are loaded")
            return subtitles
        else:
            print(f"Error: {response.status_code}")
            return None


def get_plain_text(video_url: str, language: str = cfg.LANGUAGE) -> Optional[str]:
    subs = get_subtitles(video_url=video_url, language=language)
    if subs is None:
        return None

    buffer = StringIO(subs)

    lines = []
    for line in webvtt.read_buffer(buffer):
        lines.extend(line.text.strip().splitlines())

    lines = list(dict.fromkeys(lines))
    plain_text = " ".join(lines)
    return plain_text


def divide_into_chunks(
    list_of_words: List[str], max_chunk_size: int
) -> List[List[str]]:
    chunks = []
    for i in range(0, len(list_of_words), max_chunk_size):
        chunks.append(list_of_words[i : i + max_chunk_size])
    return chunks


def split_text_into_chunks(text: str, max_chunk_size: int) -> List[str]:
    words = text.split()
    words_chunks = divide_into_chunks(
        list_of_words=words, max_chunk_size=max_chunk_size
    )
    text_chunks = [" ".join(chunk) for chunk in words_chunks]
    return text_chunks


def summarize_long_text(
    text: str,
    max_chunk_size: int = cfg.MAX_CHUNK_SIZE,
    person_type: str = cfg.PERSON_TYPE,
    model: str = cfg.MODEL,
    max_tokens: int = cfg.MAX_TOKENS,
    temperature: float = cfg.TEMPERATURE,
    top_p: float = cfg.TOP_P,
    frequency_penalty: float = cfg.FREQUENCY_PENALTY,
) -> str:
    chunks = split_text_into_chunks(text=text, max_chunk_size=max_chunk_size)

    summaries = []
    for chunk in chunks:
        response = openai.ChatCompletion.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            frequency_penalty=frequency_penalty,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant for text summarization.",
                },
                {
                    "role": "user",
                    "content": f"Summarize this for a {person_type}: {chunk}",
                },
            ],
        )
        summary = response["choices"][0]["message"]["content"]
        summary = summary.rsplit(".", 1)[0] + "."
        summaries.append(summary)

    combined_summary = " ".join(summaries)
    return combined_summary


def get_summary(video_url: str) -> Optional[str]:
    plain_text = get_plain_text(video_url=video_url, language=cfg.LANGUAGE)
    if plain_text is None:
        return None

    summary = summarize_long_text(
        text=plain_text,
        max_chunk_size=cfg.MAX_CHUNK_SIZE,
        person_type=cfg.PERSON_TYPE,
        model=cfg.MODEL,
        max_tokens=cfg.MAX_TOKENS,
        temperature=cfg.TEMPERATURE,
        top_p=cfg.TOP_P,
        frequency_penalty=cfg.FREQUENCY_PENALTY,
    )
    return summary


def translate_text(text: str, target_lang: str) -> str:
    translator = Translator()
    translation = translator.translate(text, dest=target_lang)
    translation = translation.text
    translation = ". ".join(translation.split("."))
    return translation


def print_text(text: str) -> None:
    """Small function for debug"""
    chunks = split_text_into_chunks(text=text, max_chunk_size=10)
    for chunk in chunks:
        print(chunk)
