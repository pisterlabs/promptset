import os
import sys
import copy
import pickle
import json
import random
import logging
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts.base import BasePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts.prompt import PromptTemplate
from concurrent.futures import ThreadPoolExecutor, as_completed
import dotenv
from langchain_core.documents import Document
import threading
from langchain.output_parsers import PydanticOutputParser
from langchain.callbacks.tracers import LoggingCallbackHandler
from langchain_core.output_parsers import StrOutputParser

from langchain_core.runnables import RunnableParallel

from langchain.callbacks.openai_info import OpenAICallbackHandler

from ..lib import lib_logging
from ..lib import lib_emaildb
from langchain.output_parsers.json import SimpleJsonOutputParser
import argparse


import time

lib_logging.setup_logging()

dotenv.load_dotenv()

logger = lib_logging.get_logger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the Gmail pipeline for a specific company.')
    parser.add_argument('company', choices=['cj', 'fc'], help='Specify the company environment ("cj" or "fc").')
    parser.add_argument("doc_type", help="Doc type to delete")
    args = parser.parse_args()

    lib_emaildb.set_company_environment(args.company.upper())
    docdb = lib_emaildb.get_docdb()
    res = docdb.get(include=['metadatas', 'documents', 'embeddings'])

    docdb_docs = list(sorted(zip(res['ids'], res['metadatas'], res['documents'], res['embeddings']), key=lambda x: x[0]))
    logging.debug(f"Loaded {len(docdb_docs)} documents, list:")

    for idx, (doc_id, metadata, doc_contents, embeddings) in enumerate(docdb_docs):
        if not metadata:
            logger.error(f"No metadata on document {doc_id} {doc_contents}")
            docdb._collection.delete(ids=[doc_id])
            continue

        if metadata.get('type') == args.doc_type:
            logger.debug(f"Deleting document {doc_id} {idx}/{len(docdb_docs)}: {metadata['type']}")
            docdb._collection.delete(ids=[doc_id])