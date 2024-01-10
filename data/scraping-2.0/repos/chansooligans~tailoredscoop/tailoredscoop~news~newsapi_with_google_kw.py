import asyncio
import datetime
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import quote

import aiohttp
import feedparser
import pymongo
from bs4 import BeautifulSoup
from tokenizers import Tokenizer

from tailoredscoop import utils
from tailoredscoop.db.init import SetupMongoDB
from tailoredscoop.documents.keywords import Keywords
from tailoredscoop.documents.process import DocumentProcessor
from tailoredscoop.documents.summarize import OpenaiSummarizer
from tailoredscoop.news.google_news.topics import GOOGLE_TOPICS
from tailoredscoop.openai_api import ChatCompletion

_no_default = object()


class RequestArticle:
    def __post_init__(self):
        self.logger = logging.getLogger("tailoredscoops.api")

    async def request_with_header(self, url: str, timeout: int = 600) -> str:
        """
        Send a GET request to the given URL with custom headers and return the response text.

        :param url: URL to send the request to.
        :return: Response text from the request.
        """
        # headers = {
        #     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        #     "Accept-Language": "en-US,en;q=0.5",
        # }
        headers = {
            "User-Agent": "python-requests/2.20.0",
            "Accept-Language": "en-US,en;q=0.5",
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, headers=headers, timeout=timeout, allow_redirects=True
                ) as response:
                    return await response.text(), str(response.url)
        except asyncio.TimeoutError:
            self.logger.error(f"Request timed out after {timeout} seconds: {url}")
            raise

    async def extract_article_content(
        self, url: str, source: str, kw: str
    ) -> Optional[str]:
        """
        Extract the article content from the given URL.

        :param url: URL of the article.
        :return: Extracted content of the article or None if failed.
        """
        ...
        try:
            response, redirect_url = await self.request_with_header(url)
        except Exception as e:
            self.logger.error(f"request failed: {kw} | {source} | {url} | {e}")
            return None, url

        soup = BeautifulSoup(response, "html.parser")
        article_tags = soup.find_all("article")
        if not article_tags:
            article_tags = soup.find_all(class_=lambda x: x and "article" in x)

        if article_tags:
            paragraphs = [
                p for article_tag in article_tags for p in article_tag.find_all("p")
            ]
        else:
            self.logger.error(f"soup parse failed: {kw} | {source} | {redirect_url}")
            return None, url
        content = "\n".join(par.text for par in paragraphs)
        return content, redirect_url


class ProcessArticle:
    def __post_init__(self):
        self.logger = logging.getLogger("tailoredscoops.api")

    @staticmethod
    def check_db_for_article(link, db):
        """
        check database if article was already queried
        """
        return db.articles.find_one({"link": link}, {"_id": 0})

    def published_at(self, article):
        return datetime.datetime.strptime(
            article["published"], "%a, %d %b %Y %H:%M:%S %Z"
        )

    def format_articles(self, url, article, article_text, url_hash, rank):
        return {
            "url": url,
            "link": article["link"],
            "published": self.published_at(article),
            "source": article["source"]["title"],
            "title": article["title"],
            "content": article_text,
            "created_at": datetime.datetime.now(),
            "query_id": url_hash,
            "rank": rank,
        }

    async def process_article(
        self,
        article: dict,
        url_hash: str,
        db: pymongo.database.Database,
        rank: int,
        kw: str,
    ) -> Optional[dict]:
        """
        Process a single news article and store it in the database.

        :param article: News article data.
        :param url_hash: Hash of the URL used for query_id.
        :param db: MongoDB database instance.
        :return: article is processed successfully, 0 otherwise.
        """

        stored_article = self.check_db_for_article(link=article["link"], db=db)

        if stored_article:
            return stored_article
        else:
            article_text, url = await self.extract_article_content(
                url=article["link"], source=article["source"]["title"], kw=kw
            )
            if not article_text:
                db.article_download_fails.update_one(
                    {"url": url}, {"$set": {"url": url}}, upsert=True
                )
            else:
                article = self.format_articles(
                    url=url,
                    article=article,
                    article_text=article_text,
                    url_hash=url_hash,
                    rank=rank,
                )
                db.articles.replace_one({"url": url}, article, upsert=True)
                return article


@dataclass
class NewsAPI(
    SetupMongoDB,
    DocumentProcessor,
    RequestArticle,
    ProcessArticle,
    Keywords,
):
    api_key: str = _no_default
    log: utils.Logger = utils.Logger()
    openai_api: ChatCompletion = ChatCompletion()

    def __post_init__(self):
        self.now = datetime.datetime.now()
        self.log.setup_logger()
        self.logger = logging.getLogger("tailoredscoops.newsapi")
        self.tokenizer = Tokenizer.from_pretrained("bert-base-uncased")
        self.openai_summarizer = OpenaiSummarizer(openai_api=self.openai_api)

    async def download(
        self,
        articles: List[dict],
        url_hash: str,
        db: pymongo.database.Database,
        kw: str,
    ) -> List[int]:
        """
        Download and process the given list of articles.

        :param articles: List of articles.
        :param url_hash: Hash of the URL used for query_id.
        :param db: MongoDB database instance.
        :return: List of processing results (article if success, 0 for failure).
        """
        tasks = []

        for i, article in enumerate(articles):
            tasks.append(
                asyncio.ensure_future(
                    self.process_article(
                        article=article, url_hash=url_hash, db=db, rank=i, kw=kw
                    )
                )
            )

        completed, _ = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)
        return [task.result() for task in completed]

    def exclude_sources(self, articles):
        return [
            x
            for x in articles
            if x["source"]["title"] not in ["The New York Times", "Bloomberg"]
        ]

    def get_hash(self, url):
        return hashlib.sha256(
            (url + self.now.strftime("%Y-%m-%d")).encode()
        ).hexdigest()

    async def request_google(
        self, db: pymongo.database.Database, url: str, kw: str
    ) -> List[dict]:
        """
        Request articles from Google News with the given URL and store them in the database.

        :param db: MongoDB database instance.
        :param url: URL to send the request to.
        :return: List of requested articles.
        """
        url_hash = self.get_hash(url=url)
        if db.articles.find_one({"query_id": url_hash}):
            self.logger.info(f"Query already requested: {url_hash}")
            return list(db.articles.find({"query_id": url_hash}).sort("created_at", -1))

        articles = feedparser.parse(url).entries[:30]
        articles = self.exclude_sources(articles)[:18]

        if articles:
            await self.download(articles, url_hash, db, kw)
            return list(db.articles.find({"query_id": url_hash}).sort("created_at", -1))
        else:
            self.logger.error("no articles")
            return []

    def create_url(self, query):
        if query in GOOGLE_TOPICS.keys():
            return f"""https://news.google.com/rss/topics/{GOOGLE_TOPICS[query]}"""
        else:
            return (
                f"""https://news.google.com/rss/search?q="{quote(query)}"%20when%3A1d"""
            )

    async def query_topic(self, query, db):
        new_q = self.get_topic(kw=query)
        if len(new_q) == 0:
            return []
        url = self.create_url("OR".join([f'"{x.strip()}"' for x in new_q.split(",")]))
        self.logger.info(f"alternate query for [{query}]; using {new_q}; url: {url}")
        articles = await self.request_google(db=db, url=url, kw=new_q)
        return articles

    async def query_alternate(self, query, db):
        new_q = self.get_similar_keywords_from_gpt(query)
        if len(new_q) == 0:
            return []
        url = self.create_url("OR".join([f'"{x.strip()}"' for x in new_q.split(",")]))
        self.logger.info(f"alternate query for [{query}]; using {new_q}; url: {url}")
        articles = await self.request_google(db=db, url=url, kw=new_q)
        return articles

    async def query_news_by_keywords(
        self, db: pymongo.database.Database, q: str = "Apples"
    ) -> List[dict]:
        """
        Query news articles by given keywords.

        :param db: MongoDB database instance.
        :param q: Keywords to query news articles.
        :return: List of news articles
        """
        results = []
        for query in q.split(","):
            query = query.lower()
            url = self.create_url(query)

            self.logger.info(f"query for [{query}]; url: {url}")
            articles = await self.request_google(db=db, url=url, kw=query)

            results += articles
            if len(results) <= 6:
                articles = await self.query_alternate(query=query, db=db)
                results += articles

            if len(results) <= 6:
                articles = await self.query_topic(query=query, db=db)
                results += articles

        return results
