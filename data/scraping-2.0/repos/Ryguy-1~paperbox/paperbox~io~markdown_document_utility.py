from langchain.schema.document import Document
from paperbox.utils import get_config
import os

config = get_config()


class MarkdownDocumentUtility(object):
    """Encapsulates working with a markdown file and Document objects."""

    def __init__(self, file_path: str) -> None:
        """
        Initialize the MarkdownDocumentUtility.

        Params:
            file_path (str): The path to the markdown file.
        """
        if file_path == "":
            raise ValueError("File path cannot be empty.")
        if not file_path.endswith(".md"):
            raise ValueError(f"File must be a markdown file: {file_path}")
        self.file_path = os.path.join(config["dirs"]["markdown"], file_path)
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"File not found: {self.file_path}")
        self.loaded_documents = []
        self.load_from_disk()

    def load_from_disk(self):
        """
        ReLoad the markdown file into self.loaded_documents.
        Note, this method is also called by the constructor.

        Document Format:
            page_content (str): The page content.
            metadata (dict): Metadata about the document.
        """
        with open(self.file_path, "r") as f:
            lines = f.readlines()
        if len(lines) == 0:
            return  # no documents
        # skip to first header
        while not lines[0].startswith("#"):
            lines.pop(0)
        documents = []
        current_document = None
        for line in lines:
            if line.startswith("#"):  # Start of a new document
                current_document = Document(
                    page_content=line,
                )
                documents.append(current_document)
            else:
                current_document.page_content += line
        self.loaded_documents = documents

    def save_to_disk(self) -> None:
        """
        Save the current Document objects to disk.
        """
        final_string = ""
        for document in self.loaded_documents:
            final_string += document.page_content
        with open(self.file_path, "w") as f:
            f.write(final_string)

    @staticmethod
    def get_readable_header_from_document(document: Document) -> str:
        """
        Get a readable header from a document.

        Params:
            document (Document): The document to get the header from.
        """
        return document.page_content.split("\n")[0].replace("#", "").strip()
