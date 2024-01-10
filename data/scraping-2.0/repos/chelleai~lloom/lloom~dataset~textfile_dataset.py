import glob
import uuid

from chromadb.api import Collection
from langchain.text_splitter import RecursiveCharacterTextSplitter


class TextfileDataset:
    def __init__(
        self,
        source: str,
        tokens_per_document: int,
        token_overlap: int,
        collection: Collection,
        count: int = None,
    ):
        self.source = source
        self.tokens_per_document = tokens_per_document
        self.token_overlap = token_overlap
        self.collection = collection
        self.count = count

    def load(self):
        file_paths = glob.glob(self.source)
        if not file_paths:
            raise ValueError("No files found matching the provided glob pattern.")

        ids = []
        for file_path in file_paths:
            with open(file_path, "r") as file:
                document_text = file.read()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.tokens_per_document,
                chunk_overlap=self.token_overlap,
                length_function=len,
                add_start_index=True,
            )
            document_texts = splitter.split_text(
                text=document_text,
            )

            ids += [str(uuid.uuid4()) for _ in range(len(document_texts))]

            self.collection.add(ids=ids, documents=document_texts)

        return ids
