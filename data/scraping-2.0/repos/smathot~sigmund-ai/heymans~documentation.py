import json
from pathlib import Path
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
import logging
from . import config
from . import prompt
logger = logging.getLogger('heymans')


class Documentation:
    
    def __init__(self, heymans, sources=[]):
        self._heymans = heymans
        self._documents = []
        self._sources = sources
        
    def __str__(self):
        if not self._documents:
            return ''
        return '\n\n'.join(
            f"<document>{doc.page_content}</document>" for doc in self._documents)
        
    def to_json(self):
        return json.dumps([{'page_content': doc.page_content,
                            'url': doc.metadata.get('url', None)}
                           for doc in self])
        
    def __iter__(self):
        return (doc for doc in self._documents)
    
    def __len__(self):
        return sum(len(doc.page_content) for doc in self._documents)
        
    def __contains__(self, doc):
        return doc in self._documents
    
    def prompt(self):
        if not self._documents:
            return None
        return f'''# Documentation

You have retrieved the following documentation to answer the user's question:

<documentation>
{str(self)}
</documentation>'''
    
    def append(self, doc):
        if any(doc.page_content == d.page_content for d in self):
            return
        self._documents.append(doc)
            
    def strip_irrelevant(self, question):
        important = [doc for doc in self
                     if doc.metadata.get('important', False)]
        optional = [doc for doc in self
                    if not doc.metadata.get('important', False)]
        prompts = [prompt.render(prompt.JUDGE_RELEVANCE,
                                 documentation=doc.page_content,
                                 question=question)
                   for doc in optional]
        replies = self._heymans.condense_model.predict_multiple(prompts)
        for reply, doc in zip(replies, optional):
            if not reply.lower().startswith('no'):
                important.append(doc)
            else:
                logger.info(f'stripping irrelevant documentation')
        self._documents = important
    
    def clear(self):
        logger.info('clearing documentation')
        self._documents = []
        
    def search(self, queries):
        for source in self._sources:
            for doc in source.search(queries):
                if doc not in self._documents:
                    self._documents.append(doc)
        

class BaseDocumentationSource:
    
    def __init__(self, heymans):
        self._heymans = heymans
    
    def search(self, queries):
        pass

    
class FAISSDocumentationSource(BaseDocumentationSource):
    
    def __init__(self, heymans):
        super().__init__(heymans)
        self._embeddings_model = OpenAIEmbeddings(
            openai_api_key=config.openai_api_key)
        logger.info('reading FAISS documentation cache')
        self._db = FAISS.load_local(Path('.db.cache'), self._embeddings_model)
        self._retriever = self._db.as_retriever()
    
    def search(self, queries):
        docs = []
        for query in queries:
            logger.info(f'retrieving from FAISS: {query}')
            for doc in self._retriever.invoke(query):
                if doc.page_content not in self._heymans.documentation and \
                        doc.page_content not in docs:
                    logger.info(
                        f'Retrieving {doc.metadata["url"]} (length={len(doc.page_content)})')
                    docs.append(doc)
                    break
        return docs
