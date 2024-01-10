from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from pathlib import Path
import logging
import openai
import time
from . import config
logger = logging.getLogger('heymans')


def load_library(force_reindex=False):
    db_cache = Path('.db.cache')
    src_path = Path('sources')
    embeddings_model = OpenAIEmbeddings(openai_api_key=config.openai_api_key)
    if not force_reindex and db_cache.exists():
        logger.info('loading library from cache')
        db = FAISS.load_local(db_cache, embeddings_model)
    else:
        from langchain.document_loaders import TextLoader, PyPDFLoader, \
            JSONLoader
        logger.info('initializing library')
        data = []
        # PDF files are unstructured. They can be named through config.sources
        for src in src_path.glob('pdf/**/*.pdf'):
            logger.info(f'indexing pdf: {src}')
            data += PyPDFLoader(str(src)).load_and_split()
        # jsonl is mainly for documentation
        for src in src_path.glob('jsonl/*.jsonl'):
            logger.info(f'indexing json: {src}')
            loader = JSONLoader(src, jq_schema='', content_key='content',
                                json_lines=True,
                                metadata_func=_extract_metadata)
            data += loader.load()
        # To avoid running into rate limits, we throttle the ingestion of the
        # documents
        for i in range(0, len(data), config.chunk_size):
            logger.info(
                f'ingesting chunk {i}-{i + config.chunk_size}/{len(data)}')
            chunk = data[i:i + config.chunk_size]
            if not i:
                db = FAISS.from_documents(chunk, embeddings_model)
            else:
                time.sleep(config.chunk_throttle)
                db.add_documents(chunk)
        db.save_local(db_cache)
        logger.info(f'libary initialized')
    return db.as_retriever()


def _extract_metadata(record, metadata):
    metadata['url'] = record.get('url', record.get('source', None))
    metadata['title'] = record['title']
    return metadata
