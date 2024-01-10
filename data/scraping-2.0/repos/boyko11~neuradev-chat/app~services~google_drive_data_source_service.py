import logging
from typing import List

from langchain_community.document_loaders import GoogleDriveLoader
from langchain_core.documents import Document

from services.abstract_source_list_doc_loader import AbstractSourceListDocLoader

logger = logging.getLogger(__name__)


class GoogleDriveDataSourceService(AbstractSourceListDocLoader):

    @staticmethod
    def load_docs_from_source_list(gdrive_folder_ids_list: List[str]) -> List[Document]:

        docs = []
        for gdrive_folder_id in gdrive_folder_ids_list:
            loader = GoogleDriveLoader(
                folder_id=gdrive_folder_id,
                recursive=True,
            )
            this_folder_docs = loader.load()
            docs += this_folder_docs
            logger.info(f'gdrive_folder_id: {gdrive_folder_id}, num of docs: {len(this_folder_docs)}')

        logger.info(f'Num of total google drive docs: {len(docs)}')
        return docs
