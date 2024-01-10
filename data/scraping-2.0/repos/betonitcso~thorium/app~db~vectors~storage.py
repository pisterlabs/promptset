from app.db.utils.pinecone import index
from app.schemas.vectors.embedding import OpenAIEmbedding


def insert_vectors(document_id: str, vectors: OpenAIEmbedding, tenant: str):
    embedding_vectors = [{
        'id': f'{document_id}/{i}',
        'values': vector.embedding,
        'metadata': {
            'document_id': document_id,
            'chunk_id': i,
            'tenant': tenant
        }
    } for i, vector in
        enumerate(vectors.data)]

    index.upsert(vectors=embedding_vectors)


def delete_document_vectors(document_id: str):
    index.delete(filter={
        'document_id': document_id
    })

