import logging
import os

from dotenv import load_dotenv
from google.cloud import firestore, storage
from google.cloud.firestore_v1.base_query import FieldFilter
from langchain.document_loaders import GCSFileLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Milvus
from pymilvus import MilvusClient

load_dotenv()  # take environment variables from .env.
logger = logging.getLogger(__name__)

class VectorStoreService:
    """
    A service that retrieves text data from Google Cloud Storage and feeds it into a Milvus database.
    """
    def __init__(self, run_id, project_name, bucket_name, collection_name, milvus_collection_name):
        """
        Initializes the service with the given project name and bucket name.

        :param project_name: The name of the GCP project.
        :param bucket_name: The name of the GCS bucket containing the text data.
        """
        self.run_id = run_id
        self.project_name = project_name
        self.bucket_name = bucket_name
        self.collection_name = collection_name
        self.milvus_collection_name = milvus_collection_name
        self.openai_api_key = os.getenv('OPENAI_API_KEY')
        self.milvus_api_key = os.getenv('MILVUS_API_KEY')

        self.storage_client = storage.Client()
        self.db = firestore.Client()

        self.connection_args = {
            "uri": "https://in03-5052868020ac71b.api.gcp-us-west1.zillizcloud.com",
            "user": "vaclav@pechtor.ch",
            "token": self.milvus_api_key,
            "secure": True
        }

        self.client = MilvusClient(
            uri="https://in03-5052868020ac71b.api.gcp-us-west1.zillizcloud.com",
            token=self.milvus_api_key
        )
        logger.info(f'Milvus connection: {self.client}')

        self.embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        logger.info(f'OpenAI embedings: {self.embeddings}')

        logger.info(f'Init completed. Milvus db: {self.milvus_collection_name}, Firestore db: {self.collection_name}')


    def run(self, num_docs=None):
        """
        Runs the service, processing each document in the bucket individually.

        :param num_docs: The number of documents to process. If None, all documents will be processed.
        :param collection_name: The name of the collection to store the vector data in. Defaults to 'default'.
        """
        logger.info(f'Starting VectorStoreService. Run ID: {self.run_id}')

        # Fetch file names from Firestore instead of directly from GCS
        file_names = self._get_text_filenames()

        if num_docs is not None:
            file_names = file_names[:num_docs]

        batch_size = 100
        batch_docs = []
        text_splitter = CharacterTextSplitter(chunk_size=1024, chunk_overlap=0)

        for i, file_name in enumerate(file_names):
            logger.info(f'Processing document {i}.')
            try:
                # Load the file from GCS using the file name
                loader = GCSFileLoader(project_name=self.project_name, bucket=self.bucket_name, blob=file_name)
                doc = loader.load()
                logger.info(f'Loaded document {i}.')

                docs = text_splitter.split_documents(doc)

                batch_docs.extend(docs)

                if (i + 1) % batch_size == 0:
                    logger.info('Writing batch to Milvus.')
                    _ = Milvus.from_documents(
                        batch_docs,  # process a batch of documents
                        embedding=self.embeddings,
                        connection_args=self.connection_args,
                        collection_name=self.milvus_collection_name  # Use the given collection name
                    )
                    self.client.flush(collection_name=self.milvus_collection_name)
                    num_entities = self.client.num_entities(collection_name=self.milvus_collection_name)
                    logger.info(f'Number of vectors in the database: {num_entities}')
                    batch_docs = []

                # Update the status in Firestore after each file is processed
                self._set_status_to_db_inserted(file_name)
            except Exception as e:  # pylint: disable=W0718
                logger.error(f'Exception occurred while processing document {i}: {e}', exc_info=True)

        # If there are any documents left in the batch, process them
        logger.info(f'Writing {len(batch_docs)} remaining batch_docs to Milvus.')
        if batch_docs:
            _ = Milvus.from_documents(
                batch_docs,  # process the remaining documents
                embedding=self.embeddings,
                connection_args=self.connection_args,
                collection_name=self.milvus_collection_name  # Use the given collection name
            )
            self.client.flush(collection_name=self.milvus_collection_name)
        num_entities = self.client.num_entities(collection_name=self.milvus_collection_name)
        logger.info(f'Number of vectors in the database: {num_entities}')
        logger.info('VectorStoreService has finished processing.')


    def _get_text_filenames(self):
        """
        Get all filenames of texts with status 'scraped' from Firestore.

        Returns:
        A list of filenames (str).
        """
        # Use the locally initialized client to get the collection
        collection_ref = self.db.collection(self.collection_name)

        # Define the FieldFilters
        status_filter = FieldFilter(u'status', u'==', 'scraped')
        content_type_filter = FieldFilter(u'content_type', u'==', 'text')
        file_name_filter = FieldFilter(u'file_name', u'!=', 'None')

        # Query for documents where status is 'scraped', content_type is 'text', and file_name is not 'None'
        query = collection_ref.where(filter=status_filter)
        query = query.where(filter=content_type_filter)
        query = query.where(filter=file_name_filter)

        # Execute the query and get the documents
        docs = query.stream()

        # Extract the file names from the documents
        file_names = [doc.get(u'file_name') for doc in docs]

        return file_names


    def _set_status_to_db_inserted(self, file_name):
        """
        Update the status of a document in Firestore to 'db_inserted'.

        Parameters:
        file_name (str): The name of the file stored in GCS.
        """
        # Use the locally initialized client to get the collection
        collection_ref = self.db.collection(self.collection_name)

        # Query for the document where file_name matches the given file_name
        docs = collection_ref.where(u'file_name', u'==', file_name).stream()

        # There should only be one document that matches, but we'll use a loop just in case
        for doc in docs:
            # Get a reference to the document
            doc_ref = collection_ref.document(doc.id)

            # Update the status to 'db_inserted'
            doc_ref.update({u'status': 'db_inserted'})

        logger.info(f"Updated status to 'db_inserted' for file: {file_name}")
