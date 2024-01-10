"""Fetches news articles from RSS feeds."""
import feedparser
from itertools import cycle, islice
from requests.exceptions import RequestException
import requests

from utils.logger import get_logger
from data_retrieval.openai_integration import prompt_gpt4_turbo
from config.cfg import OPENAI_API_KEY, NEWS_CONTEXT


def fetch_news(
    rss_urls: dict[str, list[str]], max_articles_per_category: int = 4
) -> str:
    """
    Fetches a limited number of news articles from multiple RSS feed URLs, alternating between sources, and converts them to natural language.

    :param rss_urls: A dictionary of RSS feed URLs, with the key being the category and the value being a list of URLs.
    :param max_articles_per_category: The maximum number of articles to fetch per category.
    :return: Natural language news articles.
    """

    markdown_output = ""
    logger = get_logger()
    logger.info("Starting to fetch news articles.")

    for category, urls in rss_urls.items():
        logger.info(f"Fetching news articles for {category}.")
        markdown_output += f"## {category}\n\n"
        seen_titles = set()
        articles_count = 0
        total_failures = 0

        url_cycle = cycle(urls)
        attempts_per_url = {url: 0 for url in urls}
        max_attempts = 3  # Maximum attempts per URL
        max_total_failures = len(urls) * max_attempts

        while articles_count < max_articles_per_category:
            url = next(url_cycle)
            attempts_per_url[url] += 1
            if attempts_per_url[url] > max_attempts:
                total_failures += 1
                if total_failures >= max_total_failures:
                    logger.warning("All URLs have reached maximum attempts. Exiting.")
                    break
                logger.warning(
                    f"Max attempts exceeded for URL {url}. Moving to the next URL."
                )
                continue

            logger.info(f"Fetching news from {url}.")
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                feed = feedparser.parse(response.content)
                if feed.bozo:
                    logger.warning(f"Invalid feed structure from {url}.")
                    continue

                entries = (
                    entry for entry in feed.entries if entry.title not in seen_titles
                )
                entry = next(islice(entries, 1), None)

                if entry:
                    title = entry.title
                    link = entry.link
                    markdown_output += f"- [{title}]({link})\n"
                    seen_titles.add(title)
                    articles_count += 1
                    logger.info(f"Fetched news from {url}.")
            except RequestException as e:
                logger.error(f"An error occurred while accessing {url}: {e}")
                continue
            except Exception as e:
                logger.error(f"An error occurred while parsing news from {url}: {e}")
                continue

        if total_failures >= max_total_failures:
            break

        logger.info(f"Fetched {articles_count} news articles for {category}.")
        markdown_output += "\n"

    logger.info("Converting news articles to natural language.")
    natural_language_news = prompt_gpt4_turbo(
        OPENAI_API_KEY, markdown_output, NEWS_CONTEXT
    )

    return natural_language_news
