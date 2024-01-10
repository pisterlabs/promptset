'''
A very simple chat bot using Claude LLM hosted on AWS bedrock with my eda markdown files.
It uses open-search for vector store.
'''
import os, logging
import sys
from langchain.document_loaders import UnstructuredMarkdownLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import MarkdownHeaderTextSplitter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("demo")

def loadSourceDocumentsAsChunks(srcdir):
    '''
    Load a source directory with all md files in it. and split it into chunks.
    '''
    logger.info("--- Load document, split ---")
    loader = DirectoryLoader(srcdir, glob="**/*.md")
    documents=loader.load()  # text and metadata
    print(documents[0])
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    all_splits = []
    for document in documents:
        all_splits +=markdown_splitter.split_text(document)
    return all_splits



if __name__ == "__main__":
    splits=loadSourceDocumentsAsChunks("../docs")
    print(splits)