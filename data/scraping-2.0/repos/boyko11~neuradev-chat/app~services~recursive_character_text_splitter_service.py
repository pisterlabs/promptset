import logging
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from services.abstract_text_splitter_service import AbstractTextSplitterService

logger = logging.getLogger(__name__)


class RecursiveCharacterTextSplitterService(AbstractTextSplitterService):

    def split_docs(self, docs: List[Document]) -> List[Document]:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=300,
            length_function=len,
            add_start_index=True,

        )

        splits = text_splitter.split_documents(docs)

        logging.info(f'Num of docs: {len(docs)}')
        logging.info(f'Num of splits: {len(splits)}')

        return splits
