import extract_msg
from typing import List, Optional

from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document


class MsgLoader(BaseLoader):
    """Load Outlook .msg file.

    Args:
        file_path: Path to the .msg file to load.
    """

    def __init__(self, file_path: str):
        """Initialize with file path."""
        self.file_path = file_path

    def load(self) -> List[Document]:
        """Load from file path."""
        msg = extract_msg.Message(self.file_path)

        msg_sender = msg.sender
        msg_sender=(msg_sender if msg_sender else '')
        msg_message = (msg_sender if msg_sender else '') + '\n'+msg.body

        msg.close()

        metadata = {"source": self.file_path, "filename": msg.filename}
        return [Document(page_content=msg_message, metadata=metadata)]
