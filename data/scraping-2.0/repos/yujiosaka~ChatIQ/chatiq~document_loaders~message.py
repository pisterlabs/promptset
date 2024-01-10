import logging
from datetime import datetime, timezone
from typing import List

from langchain.docstore.document import Document
from langchain.document_loaders.base import BaseLoader
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from chatiq.text_processor import TextProcessor
from chatiq.utils import pretty_json_dumps

from .pdf import PdfLoader
from .plain_text import PlainTextLoader
from .slack_link import SlackLinkLoader
from .unfurlink_link import UnfurlingLinkLoader


class MessageLoader(BaseLoader):
    """A class for loading Slack messages into Document objects.

    The MessageLoader class is a specialized document loader that takes a Slack message
    and loads it into a Document object.
    """

    def __init__(self, client: WebClient, body: dict, message: dict, model: str):
        """Initializes the MessageLoader with web client and a Slack body.

        Args:
            client (WebClient): The Slack WebClient to use for API calls.
            body (dict): The Slack body to be loaded, represented as a dictionary.
            message (dict): The current message in the event body.
            model (str): The name of the model used for tokenization.
        """

        self.logger = logging.getLogger(__name__)
        self.client = client
        self.body = body
        self.message = message
        self.processor = TextProcessor(model)

    def load(self) -> List[Document]:
        """Loads the Slack event into a Document object.

        Returns:
            List[Document]: A list containing a single Document object.
        """

        metadata = self._build_metadata()
        message = self.processor.truncate_text(self.message["text"])

        page_content = {
            "content_type": "message",
            "user": self.message["user"],
            "channel": self.body["event"]["channel"],
            "message": message,
            "permalink": metadata["permalink"],
            "timestamp": metadata["timestamp"],
        }

        unfurling_links = self._extract_unfurling_links()
        if unfurling_links:
            page_content["unfurling_links"] = unfurling_links

        slack_links = self._extract_slack_links()
        if slack_links:
            page_content["slack_links"] = slack_links

        files = self._extract_files()
        if files:
            page_content["files"] = files

        return [Document(page_content=pretty_json_dumps(page_content), metadata=metadata)]

    def _build_metadata(self) -> dict:
        """Builds the metadata for the Document object.

        Returns:
            dict: A dictionary containing the metadata.

        Raises:
            ValueError: If the message's timestamp cannot be converted to a date.
        """

        permalink = self._get_permalink()

        ts = self.message["ts"]
        thread_ts = self.message.get("thread_ts", ts)
        event_time = self.body["event_time"]

        try:
            timestamp = datetime.fromtimestamp(event_time, timezone.utc)
        except ValueError:
            error_message = f"Error converting date: {event_time}"
            self.logger.error(error_message)
            raise ValueError(error_message)

        return {
            "file_or_attachment_id": "",
            "content_type": "message",
            "channel_type": self.body["event"]["channel_type"],
            "channel_id": self.body["event"]["channel"],
            "thread_ts": thread_ts,
            "ts": ts,
            "timestamp": timestamp.isoformat(),
            "permalink": permalink,
        }

    def _extract_unfurling_links(self) -> List[dict]:
        """Extracts a list of unfurling link details in the Document object.

        Returns:
            List[dict]: A list of dictionaries containing the unfurling link details.
        """

        if not self.message.get("attachments"):
            return []

        return [
            {
                "title": self.processor.truncate_text(attachment["title"], length=100),
                "permalink": self.processor.truncate_text(attachment["original_url"], length=100),
            }
            for attachment in self.message["attachments"]
            if UnfurlingLinkLoader.check_supported(attachment)
        ]

    def _extract_slack_links(self) -> List[dict]:
        """Extracts a list of slack link details in the Document object.

        Returns:
            List[dict]: A list of dictionaries containing the slack link details.
        """

        if not self.message.get("attachments"):
            return []

        return [
            {
                "author": attachment["author_id"],
                "content": self.processor.truncate_text(attachment["text"], length=100),
                "permalink": self.processor.truncate_text(attachment["original_url"], length=100),
            }
            for attachment in self.message["attachments"]
            if SlackLinkLoader.check_supported(attachment)
        ]

    def _extract_files(self) -> List[dict]:
        """Extracts a list of file details in the Document object.

        Returns:
            List[dict]: A list of dictionaries containing the file details.
        """

        if not self.message.get("files"):
            return []

        return [
            {"title": file["title"], "permalink": file["permalink"]}
            for file in self.message["files"]
            if PlainTextLoader.check_supported(file) or PdfLoader.check_supported(file)
        ]

    def _get_permalink(self) -> str:
        """Retrieves the permalink to a given Slack message.

        Returns:
            str: The permalink to the message.
        """

        self.logger.debug(f"Getting permalink on team: {self.body['team_id']}")

        try:
            response = self.client.chat_getPermalink(channel=self.body["event"]["channel"], message_ts=self.message["ts"])
            return response["permalink"]
        except SlackApiError as e:
            self.logger.error(f"Failed to get permalink on team: {self.body['team_id']}. Error: {e}")
            raise
