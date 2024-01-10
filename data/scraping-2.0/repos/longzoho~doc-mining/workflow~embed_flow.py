import json
import logging
import os

from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from prefect import task, flow
from data_types import FileObj, FileStatus
from repository.profile import Profile
from util.file_util import get_file_path_by_key, save_file, file_exists, get_file_content
from util.path_util import content_path, bucket, document_path, embeddingdb_path
from langchain.document_loaders import PDFMinerLoader
from langchain.schema import Document

logger = logging.getLogger(__name__)


@task
def list_profile_file(profile_id: str) -> list[FileObj]:
    profile = Profile(profile_id=profile_id).get_profile()
    return list(
        FileObj(**file) for file in
        filter(lambda file: file.get('file_status') != FileStatus.ERROR, profile.get('files')))


MAP_LOADER = {
    '.pdf': PDFMinerLoader,
}


@task
def convert_to_document(file: FileObj, profile_id: str) -> str:
    file_key = f'{content_path()}/{file.file_hash}'
    file_extension = os.path.splitext(file_key)[1]
    file_path = get_file_path_by_key(bucket=bucket(), file_key=file_key)
    profile = Profile(profile_id=profile_id)
    loader = MAP_LOADER.get(file_extension)
    if loader is None:
        profile.update_file_status(file_hash=file.file_hash, file_status=FileStatus.ERROR)
        logger.error(f'No loader found for file extension {file_extension}')
        return None
    try:
        file_save_path = f'{document_path()}/{file.file_hash}.json'
        if not file_exists(bucket=bucket(), file_key=file_save_path):
            loader = loader(file_path=file_path)
            document = loader.load()[0]
            save_file(bucket=bucket(), file_key=file_save_path, file_data=document.json())
        profile.update_file_status(file_hash=file.file_hash, file_status=FileStatus.DOCUMENT_SAVED)
        return file_save_path
    except Exception as e:
        profile.update_file_status(file_hash=file.file_hash, file_status=FileStatus.ERROR)
        logger.error(f'Error loading file {file_path}: {e}')
        return None


@task
def remove_none(data: list[str]) -> list[str]:
    return list(filter(lambda document: document is not None, data))


chunk_size = 1000
chunk_overlap = 200


@task
def split_chunks(file_paths: list[str]) -> list[Document]:
    documents = [Document(**json.loads(get_file_content(bucket=bucket(), file_key=file_path))) for file_path in
                 file_paths]
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(documents=documents)


@task
def embed_chunk(profile_id: str, chunks: list[Document]) -> str:
    # concat all chunks to one list
    # embedd document into chroma db
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large",
        model_kwargs={"device": 'cuda'},
    )

    # embedd document into chroma db
    db = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=get_file_path_by_key(bucket=bucket(), file_key=f'{embeddingdb_path()}/{profile_id}'))
    db.persist()
    return True


@flow
def embed_flow(profile_id: str):
    files = list_profile_file(profile_id=profile_id)
    file_paths = convert_to_document.map(file=files, profile_id=profile_id)
    file_paths = remove_none(data=file_paths)
    chunks = split_chunks(file_paths=file_paths)
    return embed_chunk(profile_id=profile_id, chunks=chunks)
