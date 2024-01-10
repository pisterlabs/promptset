import uuid

import dacite
from langchain.schema import Document

from source_consumer.models.chunk_metadata import ChunkMetadata
from source_consumer.models.ingest_message import IngestMessage
from source_consumer.service.collection_strategy import CollectionStrategy
from source_consumer.service.pipeline_orchestrator import PipelineOrchestrator
from source_consumer.service.vector_orchestrator import VectorOrchestrator


class SourceIngestService:

    @staticmethod
    def process_message(key, message):
        ingest_message = dacite.from_dict(data_class=IngestMessage, data=message)

        po = PipelineOrchestrator(ingest_message)
        file_url, file_name = po.get_downloader().download_file(ingest_message)
        elements = po.get_partitioner().partition(ingest_message, file_url)

        cleaner = po.get_cleaner()
        cleaned_elements = cleaner.clean_document_pre_chunking(ingest_message, elements)
        chunked_elements = po.get_chunker().chunk_document(ingest_message, cleaned_elements)
        cleaned_chunks = cleaner.clean_document_post_chunking(ingest_message, chunked_elements)

        vo = VectorOrchestrator()
        collection_name = CollectionStrategy().get_collection_name_by_tenant(ingest_message.tenant)
        documents = SourceIngestService.prepare_data(ingest_message, file_name, cleaned_chunks)

        embedding_model, vector_size = vo.get_embeddings()
        vector_store = vo.get_vector_store(embedding_model, vector_size)
        vector_store.update_data(collection_name, documents)

        return True

    @staticmethod
    def prepare_data(ingest_message: IngestMessage, file_name: str, chunks):
        documents = []
        chunk_group_id = str(uuid.uuid4())

        for chunk_number, chunk in enumerate(chunks):
            chunk_metadata = ChunkMetadata(
                tenant_id=ingest_message.tenant,
                cluster_id=ingest_message.knowledge,
                data_folder_id=ingest_message.id,
                file_name=file_name,
                chunk_id=chunk.id,
                chunk_group_id=chunk_group_id,
                chunk_order=chunk_number,
                cloud_url=""
            )

            document = Document(page_content=chunk.text, metadata=chunk_metadata.__dict__)

            documents.append(document)

        return documents
