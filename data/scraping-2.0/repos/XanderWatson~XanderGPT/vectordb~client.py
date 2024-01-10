import pinecone
from langchain.vectorstores import Pinecone
from pinecone.core.client.exceptions import ApiException

from utils import logger

from .constants import (BAD_REQUEST, NOT_FOUND,
                        PINECONE_DEFAULT_INDEX_DIMENSIONS,
                        PINECONE_DEFAULT_METRIC, PINECONE_DEFAULT_POD_TYPE,
                        PINECONE_LIMIT_EXCEEDED_MSG)


class PineconeClient:
    def __init__(self, pinecone_api_key, pinecone_env):
        from .utils import disable_ssl_warning

        self.pinecone_api_key = pinecone_api_key
        self.pinecone_env = pinecone_env
        self.limit_exceeded = False
        self.existing_index = ""

        try:
            pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
        except Exception:
            disable_ssl_warning()
            pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)

        self.Pinecone = Pinecone

        logger.info("Pinecone client initialized successfully")

    def get_index(self, index_name):
        return pinecone.Index(index_name=index_name)

    def create_index(self, name, dimension=PINECONE_DEFAULT_INDEX_DIMENSIONS):
        index_created = False

        logger.info("Inside the create_index method of PineconeClient")

        try:
            pinecone.create_index(
                name=name,
                metric=PINECONE_DEFAULT_METRIC,
                dimension=dimension,
                pod_type=PINECONE_DEFAULT_POD_TYPE
            )

            index_created = True
        except ApiException as ex:
            if ex.status == BAD_REQUEST and \
                    ex.body == PINECONE_LIMIT_EXCEEDED_MSG:
                self.limit_exceeded = True

        if self.limit_exceeded:
            existing_indexes = pinecone.list_indexes()

            if existing_indexes:
                self.existing_index = existing_indexes[0]
                self.existing_index

        if self.existing_index:
            try:
                pinecone.delete_index(name=self.existing_index)

                index_deleted_successsfully = True
            except ApiException as ex:
                if ex.status == NOT_FOUND:
                    logger.error(
                        f"Error while deleting the existing index {ex}"
                    )

                    index_created = False

            if index_deleted_successsfully:
                try:
                    pinecone.create_index(
                        name=name,
                        metric=PINECONE_DEFAULT_METRIC,
                        dimension=dimension,
                        pod_type=PINECONE_DEFAULT_POD_TYPE
                    )

                    index_created = True
                except Exception as ex:
                    logger.error(
                        f"The error during the vector index creation was: {ex}"
                    )

                    index_created = False

        if index_created:
            return True, name
        else:
            return False, ""

    def list_indexes(self):
        return pinecone.list_indexes()

    def index_exists(self, index_name):
        return index_name in self.list_indexes()
