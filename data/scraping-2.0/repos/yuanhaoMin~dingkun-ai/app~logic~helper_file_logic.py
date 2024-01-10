import os
import shutil
import typing
import chardet
from app.config.milvus_db import MILVUS_COLLECTION, get_milvus_client
from app.constant.path_constants import DATA_DIRECTORY_PATH
from app.model.pydantic_schema.helper_schemas import GetAllFilenamesResponse
from app.util.embeddings_util import get_embeddings_with_backoff
from app.util.file_util import extract_and_remove_blank_lines
from datetime import datetime
from fastapi import UploadFile
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


def get_all_filenames_from_data_directory(user_id: int) -> GetAllFilenamesResponse:
    folder_path = DATA_DIRECTORY_PATH
    if not os.path.exists(folder_path):
        return []
    local_files = os.listdir(folder_path)
    local_filenames = {f for f in local_files if os.path.isfile(os.path.join(folder_path, f))}
    cloud_files = get_milvus_client().query(
        collection_name=MILVUS_COLLECTION,
        filter=f"(filename != '')",
        output_fields=["filename"],
    )
    cloud_filenames = {item["filename"] for item in cloud_files if "filename" in item}
    return GetAllFilenamesResponse(filenames=local_filenames.union(cloud_filenames))


def process_and_persist_business_file(
        uploadFile: UploadFile,
        user_id: int,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
) -> None:
    doc = _create_document_from_file(uploadFile, user_id)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    splitted_docs = text_splitter.split_documents([doc])

    text_list = [document.page_content for document in splitted_docs]
    metadata_list = [document.metadata for document in splitted_docs]
    vector_list = get_embeddings_with_backoff(text_list)

    client = get_milvus_client()

    filename = metadata_list[0]["filename"]
    existing_entities = client.query(
        collection_name=MILVUS_COLLECTION,
        filter=f'filename == "{filename}"',
        output_fields=["id"]
    )

    if existing_entities:
        existing_ids = [entity["id"] for entity in existing_entities]
        client.delete(
            collection_name=MILVUS_COLLECTION,
            pks=existing_ids
        )

    list_of_rows = [
        {
            "filename": metadata["filename"],
            "text": text,
            "vector": vector,
            "created_by": metadata["user_id"],
            "creation_time": metadata["creation_time"],
        }
        for text, metadata, vector in zip(text_list, metadata_list, vector_list)
    ]
    client.insert(MILVUS_COLLECTION, list_of_rows)


def purge_data_directory() -> None:
    shutil.rmtree(DATA_DIRECTORY_PATH)
    os.mkdir(DATA_DIRECTORY_PATH)


def store_file_in_data_directory(filename: str, file: typing.BinaryIO) -> None:
    file_path = os.path.join(DATA_DIRECTORY_PATH, filename)
    content = file.read()
    encoding = chardet.detect(content)['encoding']
    if encoding.lower() != 'utf-8':
        try:
            decoded_content = content.decode(encoding)
            content = decoded_content.encode('utf-8')
        except UnicodeDecodeError as e:
            raise ValueError(f"Unable to decode file: {e}")

    with open(file_path, "wb") as buffer:
        buffer.write(content)


def _create_document_from_file(uploadFile: UploadFile, user_id: int) -> list[Document]:
    processed_text = extract_and_remove_blank_lines(
        filename=uploadFile.filename, file=uploadFile.file
    )
    metadata = {
        "filename": uploadFile.filename,
        "creation_time": datetime.now().replace(microsecond=0).isoformat(),
        "user_id": user_id,
    }
    return Document(page_content=processed_text, metadata=metadata)
