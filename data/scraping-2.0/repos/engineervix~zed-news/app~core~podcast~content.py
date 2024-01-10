import datetime
import logging
import sys

# import time
from typing import Callable

# import replicate
import together

# from langchain.llms import OpenAI
from pydantic import HttpUrl

from app.core.db.models import Article, Episode

# from app.core.summarization.backends.together import brief_summary
from app.core.utilities import (
    DATA_DIR,
    # OPENAI_API_KEY,
    TOGETHER_API_KEY,
    podcast_host,
    today,
    today_human_readable,
    today_iso_fmt,
)

# llm = OpenAI(temperature=0.7, openai_api_key=OPENAI_API_KEY)


def get_episode_number() -> int:
    """Returns the episode number based on the number of episodes in the database"""
    count = Episode.select().where(Episode.live == True).count()  # noqa: E712
    return count + 1


def update_article_with_summary(title: str, url: HttpUrl, date: datetime.date, summary: str):
    """Find an article by title, URL & date, and update it with the given summary"""
    article = Article.select().where((Article.title == title) & (Article.url == url) & (Article.date == date)).first()
    if article:
        article.summary = summary
        article.save()
    else:
        logging.warning(f"Could not find article with title '{title}', URL '{url}', and date '{date}'")


def create_transcript(news: list[dict[str, str]], dest: str, summarizer: Callable):
    """Create a podcast transcript from the news, using the provided summarization function
    and write it to a file

    Args:
        news (list[dict[str, str]]): A list of news articles represented as
            dictionaries, where each dictionary contains the following keys:
            - 'source': The article source.
            - 'url': The URL of the article.
            - 'title': The title of the article.
            - 'content': The content of the article. This is passed to the OpenAI API for summarization.
            - 'category': The category of the article.
        dest (str): The destination file path where the transcript will be written.
        summarizer (Callable): The function to use for summarization. This function must accept two arguments:
            - content (str): The content of the article.
            - title (str): The title of the article.

    Raises:
        - OpenAIException: If there is an issue with the OpenAI API.
        - TimeoutError: If the summarization request times out.
        - ConnectionError: If there is a network connectivity issue.
        - ValueError: If the input data is invalid or in the wrong format.
        - TypeError: If the input data is of incorrect type.

    Returns:
        None: The function writes the transcript to the specified file but does not return any value.
    """

    articles_by_source = {}

    for article in news:
        source = article["source"].replace("Zambia National Broadcasting Corporation (ZNBC)", "ZNBC")

        # If the source is not already a key in the dictionary, create a new list
        if source not in articles_by_source:
            articles_by_source[source] = []

        # Add the article to the list for the corresponding source
        articles_by_source[source].append(article)

    prompt = f"The date is {today_human_readable}. You are {podcast_host}, a fun and witty scriptwriter, content creator and podcast host. You host the Zed News Podcast - a news and current affairs podcast which runs Monday to Friday. You are given the news from various sources as indicated below. You have a microphone in front of you, your task is to present today's episode (number {get_episode_number()}), ensuring that you cover all the news from all the sources. Where necessary, consolidate any similar news items from different sources. Ensure that the podcast is presented in a logical, coherent manner so it's easy to follow. At the end, inform your audience that you are actually not human. Remember: speak in a conversational tone, leave no news item behind and do not repeat content. Do not worry about SFX, music, captions, etc. just speak directly as if you are live on air.\n\n"

    # prompt = f"The date is {today_human_readable}. {podcast_host}, a bot, hosts the Zed News Podcast - a news and current affairs podcast which runs Monday to Friday. Your task, as an accomplished content creater and comedian, is to produce content which she'll read out word-for-word as she presents today's episode (number {get_episode_number()}). Ensure that you cover all the news from all the sources, presented in a logical, coherent manner, with any similar news items from different sources appropriately consolidated. At the end, Ayanda wants her audience to know that she's actually not a human. Remember: leave no news item behind, do not repeat content and ensure your output is presented in a human-readable style.\n\n"  # noqa: W505

    metadata = f"Title: Zed News Podcast episode {get_episode_number()}\nDate: {today_human_readable}\nHost: {podcast_host}\n\n"

    content = ""
    counter = 0
    for source in articles_by_source:
        # Iterate over each article in the source
        for article in articles_by_source[source]:
            title = article["title"]
            text = article["content"]

            # if len(news) < 15:
            #     # If there are less than 15 articles, summarize each article in the usual way
            #     summary = summarizer(text, title)
            # else:
            #     summary = brief_summary(text, title)

            summary = summarizer(text, title)

            if summary.strip().startswith("Summary: "):
                summary = summary.replace("Summary: ", "")

            update_article_with_summary(title, article["url"], today, summary)

            counter += 1

            content += f"{counter}. '{title}' (source: {source})"
            content += f"\n{summary.strip()}\n\n"

    notes = prompt + "```\n" + content + "```"

    # Write the content to a file
    with open(f"{DATA_DIR}/{today_iso_fmt}_news_headlines.txt", "w") as f:
        f.write(metadata + "News Items:\n\n" + content)

    # model = "lmsys/vicuna-13b-v1.5-16k"
    model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    temperature = 0.7
    max_tokens = 4096
    together.api_key = TOGETHER_API_KEY
    output = together.Complete.create(
        prompt=notes,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    logging.info(output)

    transcript = output["output"]["choices"][0]["text"]

    if transcript.strip():
        # Write the transcript to a file
        with open(dest, "w") as f:
            f.write(transcript)
    else:
        logging.error("Transcript is empty")
        sys.exit(1)

    # data = llm(notes)
    # if data:
    #     # Write the transcript to a file
    #     with open(dest, "w") as f:
    #         f.write(data)
    # else:
    #     logging.error("Transcript is empty")
    #     sys.exit(1)

    # model = replicate.models.get("meta/llama-2-70b-chat")
    # version = model.versions.get("02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3")
    # prediction = replicate.predictions.create(
    #     version=version,
    #     input={
    #         "prompt": notes,
    #         "temperature": 0.7,
    #         "max_new_tokens": 4096,
    #     },
    # )

    # # Check if the task is complete, then get the transcript
    # while True:
    #     logging.info("Checking if Replicate Task is completed...")
    #     prediction.reload()
    #     result = prediction.status
    #     if result == "succeeded":
    #         logging.info("Woohoo! Task completed!")
    #         break
    #     prediction.wait()

    # transcript = prediction.output

    # if transcript:
    #     # Write the transcript to a file
    #     with open(dest, "w") as f:
    #         f.write(transcript)
    # else:
    #     logging.error("Transcript is empty")
    #     sys.exit(1)
