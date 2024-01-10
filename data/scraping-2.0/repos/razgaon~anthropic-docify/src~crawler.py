from dataclasses import dataclass

from typing import Optional, cast
import urllib.parse as urlparse
from abc import ABC, abstractmethod
import logging
import requests
from bs4 import BeautifulSoup, Tag
from markdownify import MarkdownConverter
from custom_types import Source, SourceType, Metadata
from langchain.document_loaders import UnstructuredURLLoader
from unstructured.cleaners.core import  clean, clean_extra_whitespace
from youtube_transcript_api import YouTubeTranscriptApi
from googleapiclient.discovery import build

from env_var import GOOGLE_API_KEY


logger = logging.getLogger(__name__)


class Crawler(ABC):
    @abstractmethod
    def generate_row(self, url) -> Source:
        """Generates a row that contains the dataclass."""
        pass


class WebpageCrawler(Crawler):
    def __init__(self, source_type: SourceType, use_unstructured=True) -> None:
        super().__init__()
        self.source_type = source_type
        self.use_unstructured = use_unstructured

    def _get_webpage_body(self, url: str) -> Tag:
        """Uses BeautifulSoup4 to fetch a webpage's HTML body given a URL"""
        response = requests.get(url)
        if response.status_code != 200:
            raise Exception(
                f"Failed to fetch the webpage. Status code: {response.status_code}"
            )
        html_content = response.text
        soup = BeautifulSoup(html_content, "html.parser")
        parent = soup.find("article")
        if not parent:
            raise Exception(f"No article tag found for url {url}")
        main_content = parent.find("div", class_="markdown")
        return cast(Tag, main_content)

    def _html_to_markdown(self, body: Tag) -> str:
        return MarkdownConverter().convert_soup(body)

    def generate_row(self, url: str) -> Source:
        logging.info("Starting webpage crawling")
        if self.use_unstructured:
            res = Source(
                url=url,
                content=self._get_unstructured_document(url),
                metadata=Metadata(
                    source_type=self.source_type,
                ),
            )
        else:
            res = Source(
                url=url,
                content=self._html_to_markdown(self._get_webpage_body(url)),
                metadata=Metadata(
                    source_type=self.source_type,
                ),
            )
        logger.info("Finished webpage crawling")
        return res

    def _get_unstructured_document(self, url):
        "Given an URL, return a langchain Document to futher processing"
        loader = UnstructuredURLLoader(
            urls=[url],
            mode="elements",
            post_processors=[clean, clean_extra_whitespace],
        )
        elements = loader.load()
        selected_elements = [
            e for e in elements if e.metadata["category"] == "NarrativeText"
        ]
        full_clean = " ".join([e.page_content for e in selected_elements])
        return full_clean


@dataclass
class YoutubeMetadata:
    title: str
    description: str
    channel_title: str
    published_at: str

class YoutubeCrawler(Crawler):

    def _get_video_id(self, video_url: str):
        """
        This function extracts the YouTube video ID from an URL.
        """

        url_data = urlparse.urlparse(video_url)
        video_id = urlparse.parse_qs(url_data.query)["v"][0]

        return video_id

    def _get_transcript(self, video_url: str) -> str:
        video_id = self._get_video_id(video_url)
        try:
            # This will return a list of dictionaries, each containing a single part of the transcript
            logger.info("Starting transcribing")
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
            logger.info("Finished transcribing")
            # Now we will combine all parts into a single transcript
            transcript = " ".join([d["text"] for d in transcript_list])
            return transcript
        except Exception as e:
            logger.error(f"Error getting transcript for video {video_url}: {e}")
            return ""

    def _get_video_metadata(self, video_url: str) -> Optional[YoutubeMetadata]:
        video_id = self._get_video_id(video_url)

        youtube = build("youtube", "v3", developerKey=GOOGLE_API_KEY)

        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()

        if response["items"]:
            item = response["items"][0]

            title = item["snippet"]["title"]
            description = item["snippet"]["description"]
            channel_title = item["snippet"]["channelTitle"]
            published_at = item["snippet"]["publishedAt"]

            return YoutubeMetadata(
                title=title,
                description=description,
                channel_title=channel_title,
                published_at=published_at,
            )

        else:
            logger.error(f"No metadata found for video: {video_url}")
            return None

    def generate_row(self, url):
        content = self._get_transcript(url)
        authors = []
        metadata = self._get_video_metadata(url)
        if metadata:
            authors.append(metadata.channel_title)
        return Source(
            url=url,
            source_type=SourceType.Youtube,
            content=content,
            authors=authors,
        )
