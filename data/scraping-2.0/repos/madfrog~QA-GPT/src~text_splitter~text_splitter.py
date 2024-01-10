# coding=utf-8

from typing import List

from langchain.schema.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from tools.logging_helper import LoggerHelper

logger = LoggerHelper().get_logger()

class TextSplitter(object):
    def split_text(self, docs: List[Document], type: str) -> List[Document]:
        '''
        @type literal string, such as txt, pdf, md
        '''
        if type == 'md':
            headers_to_split_on = [
                ('#', 'Section_1'),
                ('##', 'Section_2'),
                ("###", "Section_3"),
                ("####", "Section_4"),
                ("#####", "Section_5"),
            ]
            md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
            logger.debug(f'get markdown document len: {len(docs)}')
            docs = md_splitter.split_text(docs[0].page_content)

        logger.debug(f'get document len: {len(docs)}')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
        all_splits = text_splitter.split_documents(docs)
        return all_splits
