from typing import List
from typing import Optional

from langchain.document_loaders.base import BaseLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import TextSplitter
from loguru import logger

from maicoin_faq_bot.utils import load_json


class FAQLoader(BaseLoader):

    def load(self, json_file: str) -> List[Document]:
        logger.info(f'loading json file: {json_file}')
        data = load_json(json_file)

        logger.info('creating documents...')
        docs = []
        for d in data:
            page_content = (f'{d["title"]}\n'
                            f'{d["body"]}')
            docs.append(Document(page_content=page_content))

        return docs

    def load_and_split(self, json_file: str, text_splitter: Optional[TextSplitter] = None) -> List[Document]:
        if text_splitter is None:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000)

        docs = self.load(json_file=json_file)

        logger.info('splitting documents...')
        return text_splitter.split_documents(docs)
