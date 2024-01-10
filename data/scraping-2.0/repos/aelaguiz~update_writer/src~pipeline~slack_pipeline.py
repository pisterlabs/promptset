import sys
import dotenv
from ..lib.lib_logging import get_logger, setup_logging
from ..lib.lib_logging import get_logger, get_run_logger, setup_logging, set_console_logging_level

from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from ..lib import lib_gmail
from ..lib import lib_docdb

dotenv.load_dotenv()
setup_logging()
logger = get_logger()
COMPANY_ENV = None

import json
import zipfile
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm


from langchain_core.documents import Document

from langchain_community.document_loaders.base import BaseLoader
from itertools import islice
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class SlackDirectoryLoader(BaseLoader):
    """Load from a `Slack` directory dump."""

    def __init__(self, zip_path: str, workspace_url: Optional[str] = None):
        """Initialize the SlackDirectoryLoader.

        Args:
            zip_path (str): The path to the Slack directory dump zip file.
            workspace_url (Optional[str]): The Slack workspace URL.
              Including the URL will turn
              sources into links. Defaults to None.
        """
        self.zip_path = Path(zip_path)
        self.workspace_url = workspace_url
        self.channel_id_map = self._get_channel_id_map(self.zip_path)

    @staticmethod
    def _get_channel_id_map(zip_path: Path) -> Dict[str, str]:
        """Get a dictionary mapping channel names to their respective IDs."""
        with zipfile.ZipFile(zip_path, "r") as zip_file:
            try:
                with zip_file.open("channels.json", "r") as f:
                    channels = json.load(f)
                return {channel["name"]: channel["id"] for channel in channels}
            except KeyError:
                return {}

    def load(self) -> List[Document]:
        """Load and return documents from the Slack directory dump."""
        docs = []
        with zipfile.ZipFile(self.zip_path, "r") as zip_file:
            for channel_path in zip_file.namelist():
                channel_name = Path(channel_path).parent.name
                if not channel_name:
                    continue
                if channel_path.endswith(".json"):
                    messages = self._read_json(zip_file, channel_path)
                    for message in messages:
                        document = self._convert_message_to_document(
                            message, channel_name
                        )
                        docs.append(document)
        return docs

    def _read_json(self, zip_file: zipfile.ZipFile, file_path: str) -> List[dict]:
        """Read JSON data from a zip subfile."""
        with zip_file.open(file_path, "r") as f:
            data = json.load(f)
        return data

    def _convert_message_to_document(
        self, message: dict, channel_name: str
    ) -> Document:
        """
        Convert a message to a Document object.

        Args:
            message (dict): A message in the form of a dictionary.
            channel_name (str): The name of the channel the message belongs to.

        Returns:
            Document: A Document object representing the message.
        """
        text = message.get("text", "")
        metadata = self._get_message_metadata(message, channel_name)
        return Document(
            page_content=text,
            metadata=metadata,
        )

    def _get_message_metadata(self, message: dict, channel_name: str) -> dict:
        """Create and return metadata for a given message and channel."""
        timestamp = message.get("ts", "")
        user = message.get("user", "")

        user_profile = message.get("user_profile", {})
        avatar_hash = user_profile.get("avatar_hash", "")
        image_72 = user_profile.get("image_72", "")
        first_name = user_profile.get("first_name", "")
        real_name = user_profile.get("real_name", "")
        display_name = user_profile.get("display_name", "")
        team = user_profile.get("team", "")
        name = user_profile.get("name", "")

        is_restricted = user_profile.get("is_restricted", False)
        is_ultra_restricted = user_profile.get("is_ultra_restricted", False)
        
        source = self._get_message_source(channel_name, user, timestamp)
        return {
            "source": source,
            "slack_id": message.get("client_msg_id", ""),
            "channel": channel_name,
            "timestamp": timestamp,
            "user": user,
            "avatar_hash": avatar_hash,
            "image_72": image_72,
            "first_name": first_name,
            "real_name": real_name,
            "display_name": display_name,
            "team": team,
            "name": name,
            "is_restricted": is_restricted,
            "is_ultra_restricted": is_ultra_restricted,
            'type': 'slack'
        }

    def _get_message_source(self, channel_name: str, user: str, timestamp: str) -> str:
        """
        Get the message source as a string.

        Args:
            channel_name (str): The name of the channel the message belongs to.
            user (str): The user ID who sent the message.
            timestamp (str): The timestamp of the message.

        Returns:
            str: The message source.
        """
        if self.workspace_url:
            channel_id = self.channel_id_map.get(channel_name, "")
            return (
                f"{self.workspace_url}/archives/{channel_id}"
                + f"/p{timestamp.replace('.', '')}"
            )
        else:
            return f"{channel_name} - {user} - {timestamp}"


def process_and_load_document(docs):
    try:
        logger.debug(f"Processing and loading documents {docs}")
        lib_docdb.add_docs(docs)
        return True
    except Exception as e:
        logger.error(f"Error loading document: {e}")
        return False

def batch(iterable, size):
    """
    Split an iterable into batches of a specified size.
    :param iterable: The iterable to split into batches.
    :param size: The size of each batch.
    :return: A generator yielding batches of the specified size.
    """
    iterator = iter(iterable)
    while True:
        batch = list(islice(iterator, size))
        if not batch:
            return
        yield batch

def load_slack_data(slack_zip_path, slack_workspace_url, max_workers=10):
    """
    Load Slack data from the given zip file path and load it into the document database.
    :param slack_zip_path: Path to the Slack export zip file.
    :param slack_workspace_url: URL of the Slack workspace for proper doc sources.
    :param max_workers: Maximum number of threads for concurrent processing.
    :return: None
    """
    try:
        loader = SlackDirectoryLoader(slack_zip_path, slack_workspace_url)
        docs = loader.load()

        # loaded_slacks = list(lib_emaildb.get_slack_ids())
        loaded_slacks = []
        logger.info("Starting to load Slack documents into the database.")

        batch_size = 50
        ready_docs = [doc for doc in docs if doc.metadata['slack_id'] not in loaded_slacks]

        success_count = 0
        failure_count = 0

        for idx, batch_docs in enumerate(tqdm(batch(ready_docs, batch_size), total=len(ready_docs)//batch_size)):
            logger.info(f"Processing batch {idx} {batch_size} documents")
            result = process_and_load_document(batch_docs)
            if result:
                success_count += 1
            else:
                failure_count += 1

        # with ThreadPoolExecutor(max_workers=max_workers) as executor, tqdm(total=len(ready_docs)) as pbar:
        #     futures = []
        #     for batch_docs in batch(ready_docs, batch_size):
        #         future = executor.submit(process_and_load_document, batch_docs)
        #         futures.append(future)

        #     for future in as_completed(futures):
        #         result = future.result()
        #         if result:
        #             success_count += 1
        #         else:
        #             failure_count += 1
        #         pbar.update(batch_size)

        logger.info("Completed loading Slack documents into the database.")
        logger.info(f"Successful loads: {success_count} representing {success_count * batch_size} documents (% of total: {success_count / len(ready_docs) * 100.0})")
        logger.info(f"Failed loads: {failure_count} representing {failure_count * batch_size} documents (% of total: {failure_count / len(ready_docs) * 100.0})")

    except Exception as e:
        logger.error(f"Error loading Slack data: {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) < 3:
        logger.error("Usage: python script.py <path_to_slack_zip> <slack_workspace_url>")
        sys.exit(1)

    lib_docdb.set_company_environment("CJ")

    slack_zip_path = sys.argv[1]
    slack_workspace_url = sys.argv[2]
    nworkers = int(sys.argv[3])
    load_slack_data(slack_zip_path, slack_workspace_url, nworkers)