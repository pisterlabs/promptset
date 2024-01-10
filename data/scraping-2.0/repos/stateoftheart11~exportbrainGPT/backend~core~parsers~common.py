import time
from uuid import uuid4

from langchain.schema import Document
from models.brains import Brain
from models.files import File
from models.data import Data
from utils.vectors import Neurons
from utils.file import compute_sha1_from_content

from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer


async def process_file(
    file: File,
    loader_class,
    enable_summarization,
    brain_id,
    user_openai_api_key,
):
    dateshort = time.strftime("%Y%m%d")

    file.compute_documents(loader_class)

    encoder = SentenceTransformer('all-MiniLM-L6-v2') 
    records = []
    metadata = {
        "data_sha1": file.file_sha1,
        "data_size": file.file_size,
        "data_name": file.file_name,
        "chunk_size": file.chunk_size,
        "chunk_overlap": file.chunk_overlap,
        "date": dateshort,
        # "summarization": "true" if enable_summarization else "false",
    }
    for doc in file.documents:  # pyright: ignore reportPrivateUsage=none
        record = models.Record(
			id=str(uuid4()),
			vector=encoder.encode(doc.page_content).tolist(),
			payload={
                "data_sha1": file.file_sha1,
                "brain_id": brain_id,
                "content": doc.page_content
            }
		)
        records.append(record)

        # doc_with_metadata = Document(page_content=doc.page_content, metadata=metadata)

        # neurons = Neurons()
        # created_vector = neurons.create_vector(doc_with_metadata, user_openai_api_key)
        # # add_usage(stats_db, "embedding", "audio", metadata={"file_name": file_meta_name,"file_type": ".txt", "chunk_size": chunk_size, "chunk_overlap": chunk_overlap})

        # created_vector_id = created_vector[0]  # pyright: ignore reportPrivateUsage=none

        # brain = Brain(id=brain_id)
        # brain.create_brain_vector(created_vector_id, file.file_sha1)
    
    brain = Brain(id=brain_id)
    brain.create_brain_data(file.file_sha1, metadata)
    file.upload_records_qdrant(records)

    return


async def process_data(
    data: Data,
    brain_id,
):
    dateshort = time.strftime("%Y%m%d")

    data.compute_documents()
    encoder = SentenceTransformer('all-MiniLM-L6-v2') 

    metadata = {
        "data_sha1": data.data_sha1,
        "data_size": data.data_size,
        "data_name": data.data_name,
        "chunk_size": data.chunk_size,
        "chunk_overlap": data.chunk_overlap,
        "date": dateshort,
        # "summarization": "true" if enable_summarization else "false",
    }
    records = []
    for doc in data.documents:  # pyright: ignore reportPrivateUsage=none
        # doc_with_metadata = Document(page_content=doc.page_content, metadata=metadata)

        # neurons = Neurons()
        # created_vector = neurons.create_vector(doc_with_metadata, user_openai_api_key)
        # # add_usage(stats_db, "embedding", "audio", metadata={"file_name": file_meta_name,"file_type": ".txt", "chunk_size": chunk_size, "chunk_overlap": chunk_overlap})

        # created_vector_id = created_vector[0]  # pyright: ignore reportPrivateUsage=none
        record = models.Record(
			id=str(uuid4()),
			vector=encoder.encode(doc).tolist(),
			payload={
                "data_sha1": data.data_sha1,
                "brain_id": brain_id,
                "content": doc
            }
		)
        records.append(record)
    
    brain = Brain(id=brain_id)
    brain.create_brain_data(data.data_sha1, metadata)
    data.upload_records_qdrant(records)

    return