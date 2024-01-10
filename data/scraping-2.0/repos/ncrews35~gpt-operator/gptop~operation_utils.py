import os
import json
from uuid import uuid4
import pinecone
from openai.embeddings_utils import get_embedding
from .factory import create_operation, create_operation_from_object


__all__ = ["OperationUtils"]


class OperationUtils():
    """
    A set of utility objects for managing operations
    """

    @classmethod
    def create_operation(cls, namespace, operation_type, name, description, metadata, schema):
        """
        Creates an operation, generates an embedding from it, and
        stores it in a vector database.
        - namespace: The namespace to store the embedding
        - operation_type: The type of operation
        - metadata: The predefined metadata of the operation
        - schema: The schema of the operation

        Also writes the operation ID to the `ops_list.txt` file

        Returns: The created operation
        """

        index = pinecone.Index(os.getenv("PINECONE_INDEX"))

        operation_id = str(uuid4())

        operation = create_operation(
            id=operation_id,
            type=operation_type,
            name=name,
            description=description,
            metadata=json.dumps(json.loads(metadata), separators=(',', ': ')),
            schema=json.dumps(json.loads(schema), separators=(',', ': '))
        )

        embedding = get_embedding(operation.embedding_obj(),
                                  engine="text-embedding-ada-002")

        to_upsert = zip([operation_id], [embedding], [operation.vector_metadata()])

        index.upsert(vectors=list(to_upsert), namespace=namespace)

        return operation

    @classmethod
    def get_operation(cls, namespace: str, operation_id: str):
        """
        Fetches an existing operation from the vector database.
        - namespace: The namespace to store the embedding
        - operation_id: The identifier of the operation

        Returns: The operation represented
        """
        index = pinecone.Index(os.getenv("PINECONE_INDEX"))

        result = index.fetch([operation_id], namespace=namespace)
        vectors = result.get('vectors')
        vector = vectors.get(operation_id)

        if not vector:
            return None

        obj = vector.get('metadata')
        return create_operation_from_object(obj)

    @classmethod
    def update_operation(cls, namespace, operation_id, operation_type, name, description, metadata, schema):
        """
        Updates an existing operation, generates a new embedding, and
        overrides the existing operation in the vector database.
        - namespace: The namespace to store the embedding
        - operation_id: The identifier of the operation
        - operation_type: The type of operation
        - metadata: The predefined metadata of the operation
        - schema: The schema of the operation

        Returns: The updated operation
        """

        operation = create_operation(
            id=operation_id,
            type=operation_type,
            name=name,
            description=description,
            metadata=json.dumps(json.loads(metadata), separators=(',', ': ')),
            schema=json.dumps(json.loads(schema), separators=(',', ': '))
        )

        index = pinecone.Index(os.getenv("PINECONE_INDEX"))

        embedding = get_embedding(operation.embedding_obj(),
                                  engine="text-embedding-ada-002")

        to_upsert = zip([operation_id], [embedding], [operation.vector_metadata()])

        index.upsert(vectors=list(to_upsert), namespace=namespace)

        return operation

    @classmethod
    def remove_operation(cls, namespace: str, operation_id: str):
        """
        Removes an existing operation from the vector database.
        - namespace: The namespace to store the embedding
        - operation_id: The identifier of the operation

        Also removes the operation ID from `ops_list.txt` file.
        """

        index = pinecone.Index(os.getenv("PINECONE_INDEX"))

        index.delete(ids=[operation_id], namespace=namespace)

    @classmethod
    def remove_namespace(cls, namespace: str):
        """
        [DANGEROUS] Deletes an entire namespace of operations.
        - namespace: The namespace in the vector database
        """
        index = pinecone.Index(os.getenv("PINECONE_INDEX"))

        index.delete(deleteAll='true', namespace=namespace)