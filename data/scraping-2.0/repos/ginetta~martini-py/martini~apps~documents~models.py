import os
import logging

from django.db import models
from django.conf import settings

logger = logging.getLogger(__name__)


class DocumentCollection(models.Model):
    '''
    DocumentCollection regroups UnstructuredDocuments.
    In other words, a DocumentCollection is a group of documents that are related,
    and that can be searched together.
    DocumentCollection stores its documents in a single Qdrant collection.
    '''
    name = models.CharField(max_length=75, unique=True, null=True)
    slug = models.SlugField(unique=True, null=True)
    description = models.TextField(null=True, blank=True)

    def __str__(self):
        return f'DocumentCollection (name="{self.name}")'


class UnstructuredDocument(models.Model):
    name = models.CharField(max_length=255, default='Untitled')
    description = models.TextField(null=True, blank=True)
    file = models.FileField()
    task_id = models.CharField(max_length=255, blank=True, null=True)
    has_embeddings = models.BooleanField(default=False)
    collection = models.ForeignKey(
        DocumentCollection,
        related_name='documents',
        on_delete=models.CASCADE,
    )

    def __str__(self):
        return f'UnstructuredDocument (name="{self.name}")'

    @classmethod
    def chunk_document(cls, filepath: str, doc_name: str):
        '''
        Chunks a PDF document into smaller documents.

        Args:
            filepath (str): Path to the file to process.
            doc_name (str): Name of the document the embeddings are extracted from;
                this is used as a marker in the metadata of the embeddings for when
                the document is deleted.
        '''
        from langchain.document_loaders import PDFMinerLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter

        # For PDFs, we use PyPDF2 to load the document.
        loader = PDFMinerLoader(filepath)
        data = loader.load()
        logger.info(f'Data loaded from document {doc_name} (file path: {filepath})')

        # The Document (note the capital D as it is an instance of langchain.docstore.document.Document)
        # is milked out of its text content. The text content is then split into smaller chunks.
        chunk_size = settings.MARTINI_DEFAULT_TEXT_SPLITTER_CHUNK_SIZE
        chunk_overlap = settings.MARTINI_DEFAULT_TEXT_SPLITTER_CHUNK_OVERLAP
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        texts = text_splitter.split_text(data[0].page_content)
        return texts

    @classmethod
    def save_embeddings(
        cls,
        filepath: str = None,
        collection_name: str = None,
        doc_name: str = None,
        instance_id: int = None
    ):
        '''
        Save the embeddings of a document in Qdrant.
        Used directly in local development, and as a Celery task in production.

        Args:
            filepath (str): Path to the file to process.
            collection_name (str): Name of the collection to store the embeddings in.
            doc_name (str): Name of the document the embeddings are extracted from;
                this is used as a marker in the metadata of the embeddings for when
                the document is deleted.
            instance_id (int): ID of the UnstructuredDocument instance.
        '''
        from apps.documents.vectorstore import get_vectorstore_for_chains
        from apps.documents.exceptions import UnprocessableDocumentError

        from apps.chats.llm import get_embeddings_model

        if not collection_name or not doc_name or not instance_id:
            raise ValueError('collection_name, doc_name and instance_id must be specified')

        if not os.path.exists(filepath):
            raise ValueError(f'File {filepath} does not exist')

        texts = cls.chunk_document(filepath, doc_name)
        if len(texts) == 0:
            raise UnprocessableDocumentError(
                f'No text extracted from {doc_name}. Maybe this is an image only document?'
            )
        logger.info(f'Text from {doc_name} was split into {len(texts)} smaller chunks')
        embeddings = get_embeddings_model()
        qdrant = get_vectorstore_for_chains(embeddings, collection_name=collection_name)
        # Create a list of metadata dictionaries, one for each text, that all contains
        # the Document instance ID as metadata, to be added to the embeddings.
        # This is used to facilitate deletion of the embeddings of a removed Document.
        metadata_list = [{'instance_id': instance_id} for _ in range(len(texts))]
        qdrant.add_texts(texts, metadata_list)

        # Update the UnstructuredDocument instance to reflect that the embeddings
        # have been stored in Qdrant.
        cls.objects.filter(id=instance_id).update(
            has_embeddings=True,
            task_id=None
        )

    @classmethod
    def delete_embeddings(
        cls,
        collection_name: str,
        doc_name: str,
        instance_id: int
    ):
        '''
        Delete embeddings from Qdrant associated with a document.
        Used directly in local development, and as a Celery task in production.

        Args:
            collection_name (str): Name of the collection to store the embeddings in.
            doc_name (str): Name of the document the embeddings are extracted from;
                this is used as a marker in the metadata of the embeddings for when
                the document is deleted.
            instance_id (int): ID of the UnstructuredDocument instance.
        '''
        from apps.documents.vectorstore import delete_points_by_metadata

        if not collection_name or not doc_name or not instance_id:
            raise ValueError('collection_name, doc_name and instance_id must be specified')

        delete_points_by_metadata(collection_name, doc_name, instance_id)
        cls.objects.filter(id=instance_id).update(
            has_embeddings=False,
            task_id=None
        )
