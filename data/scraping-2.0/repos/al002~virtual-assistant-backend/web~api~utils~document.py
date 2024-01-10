from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings

from celery.exceptions import MaxRetriesExceededError
from web.celery import app

from virtual_assistant.utilities import PINECONE_INDEX_NAME

from ..models import Document
from ..models.document import IndexingStatus

def index_document(doc_id, doc_type, file_path):
    document = Document.objects.get(id=doc_id)
    document.indexing_status = IndexingStatus.INDEXING
    document.save()

    embedding = OpenAIEmbeddings()

    if doc_type == 'url':
        loader = WebBaseLoader([file_path])
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        Pinecone.from_documents(index_name=PINECONE_INDEX_NAME, embedding=embedding, documents=docs, ids=[str(doc_id)])

    document.indexing_status = IndexingStatus.SUCCESS
    document.save()

@app.task(bind=True, max_retries=3, default_retry_delay=60)
def index_document_task(self, doc_id, doc_type, file_path):
    try:
        index_document(doc_id, doc_type, file_path)
    except Exception as e:
        try:
            # Retry the task with a delay
            self.retry(countdown=self.default_retry_delay, exc=e)
        except MaxRetriesExceededError:
            # Log the error or notify the admin after reaching the maximum retry attempts
            document = Document.objects.get(id=doc_id)
            document.indexing_status = IndexingStatus.SUCCESS
            document.save()
            pass